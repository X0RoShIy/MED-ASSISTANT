"""
Дообучение Llama 3 8B методом QLoRA на медицинском датасете.

QLoRA = 4-битная квантизация базовой модели (NF4) + обучаемые LoRA-адаптеры.
Это позволяет дообучить 8B-модель на одной GPU с 8 ГБ VRAM: базовые веса
заморожены и сжаты до 4 бит, обучаются только маленькие адаптеры (~0.5% параметров).

ТРЕБОВАНИЯ (установить заранее):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers peft bitsandbytes trl datasets accelerate

ДОСТУП К МОДЕЛИ:
    Llama 3 8B требует согласия с лицензией Meta на HuggingFace.
    1) Создайте аккаунт на huggingface.co
    2) Примите условия на странице meta-llama/Meta-Llama-3-8B-Instruct
    3) huggingface-cli login   (вставьте токен)

ЗАПУСК:
    python finetune/train_qlora.py
    # или с параметрами:
    python finetune/train_qlora.py --epochs 3 --batch-size 1 --lr 2e-4

РЕЗУЛЬТАТ:
    Папка finetune/lora_adapter/ с обученным адаптером (~50-200 МБ).
    Базовые веса Llama НЕ дублируются — адаптер подключается к ним при инференсе.
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path

# ВАЖНО: глушим лишние логи TensorFlow (он подтягивается транзитивно и засоряет вывод).
# Ставим до импорта тяжёлых библиотек.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "finetune" / "dataset"
OUTPUT_DIR = ROOT / "finetune" / "lora_adapter"

# Базовая модель. Можно заменить на меньшую, если 8 ГБ не хватает:
#   "meta-llama/Meta-Llama-3-8B-Instruct"  — основная (нужен доступ Meta)
#   "NousResearch/Meta-Llama-3-8B-Instruct" — зеркало без gated-доступа
#   "Qwen/Qwen2.5-7B-Instruct"             — альтернатива, отлично знает русский
#   "Qwen/Qwen2.5-3B-Instruct"             — если 8 ГБ не хватит даже с QLoRA
DEFAULT_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="накопление градиента: эффективный batch = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--output", default=str(OUTPUT_DIR))
    ap.add_argument("--no-quant", action="store_true",
                    help="обучать БЕЗ 4-битной квантизации (обход bitsandbytes). "
                         "Подходит для моделей до ~3-4B на 8 ГБ VRAM. "
                         "Используйте, если bitsandbytes падает с 0xC0000005.")
    args = ap.parse_args()

    # Импорты внутри main, чтобы файл можно было прочитать без установленных библиотек
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    print(f"Базовая модель: {args.model}", flush=True)
    print(f"GPU доступен: {torch.cuda.is_available()}", flush=True)

    # ВАЖНО: без CUDA-сборки torch обучение невозможно — bitsandbytes 4-bit
    # не работает на CPU. Останавливаемся сразу с понятным объяснением,
    # а не падаем позже на загадочном ValueError про CPU dispatch.
    if not torch.cuda.is_available():
        raise SystemExit(
            "\n[ОШИБКА] GPU не виден (torch.cuda.is_available() == False).\n"
            "Скорее всего установлена CPU-сборка PyTorch.\n"
            "Исправление:\n"
            "  pip uninstall -y torch torchvision torchaudio\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "Проверка:  python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "Должно вывести True. Версию CUDA смотрите в nvidia-smi (нужна >= 12.1)."
        )

    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  {torch.cuda.get_device_name(0)}, {gpu_gb:.1f} ГБ", flush=True)
    if gpu_gb < 7.5:
        print("  [ПРЕДУПРЕЖДЕНИЕ] Меньше 8 ГБ VRAM. Если поймаете "
              "'CUDA out of memory', возьмите модель поменьше "
              "(--model Qwen/Qwen2.5-3B-Instruct) или --max-seq-len 768.", flush=True)

    # ВАЖНО: квантизация 8B-модели читает веса в RAM до переноса на GPU.
    # При нехватке оперативной памяти процесс молча убивается ОС (без трейсбэка).
    # Это самая частая причина «ничего не вывелось и процесс остановился».
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        print(f"  Свободно RAM: {avail_gb:.1f} ГБ", flush=True)
        if avail_gb < 12:
            print("  [ВНИМАНИЕ] Мало свободной RAM (<12 ГБ). Загрузка 8B-модели "
                  "может вызвать тихий вылет. Закройте другие программы или "
                  "используйте --model Qwen/Qwen2.5-3B-Instruct.", flush=True)
    except ImportError:
        pass

    # --- 2. Загрузка токенизатора ---
    print("Загрузка токенизатора...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Загрузка модели: два режима ---
    if args.no_quant:
        # Режим БЕЗ bitsandbytes: модель в bf16, LoRA сверху.
        # Подходит для моделей до ~3-4B на 8 ГБ VRAM (Qwen2.5-3B ≈ 6 ГБ в bf16).
        # Обходит краши bitsandbytes (0xC0000005 на Windows).
        print("Режим без квантизации (bf16). Загрузка модели...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # нужно для LoRA + checkpointing без квантизации
    else:
        # Режим QLoRA: 4-битная квантизация через bitsandbytes.
        print("Загрузка и квантизация модели в 4 бита (это самый долгий шаг, "
              "может занять несколько минут — дождитесь)...", flush=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # ВАЖНО: device_map={"": 0} кладёт всю модель на GPU 0 целиком.
        # device_map="auto" на 8 ГБ может вынести слои на CPU, а 4-битная
        # модель так не работает — отсюда ValueError про CPU dispatch.
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False  # несовместимо с gradient checkpointing

    # --- 3. Конфигурация LoRA ---
    # Обучаем только адаптеры на проекциях внимания и MLP.
    lora_config = LoraConfig(
        r=16,                    # ранг адаптера (8-32; больше = выразительнее, но тяжелее)
        lora_alpha=32,           # масштаб (обычно 2*r)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Обучаемых параметров: {trainable:,} из {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # --- 4. Датасет ---
    data_files = {
        "train": str(DATASET_DIR / "train.jsonl"),
        "validation": str(DATASET_DIR / "val.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    def format_chat(example):
        # Превращаем messages в строку через шаблон чата модели
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(format_chat)

    # --- 5. Аргументы обучения ---
    # ВАЖНО: 8-битный оптимизатор тоже из bitsandbytes — в режиме --no-quant
    # используем обычный adamw_torch, иначе будет тот же краш.
    optimizer = "adamw_torch" if args.no_quant else "paged_adamw_8bit"
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,        # экономия памяти ценой скорости
        optim=optimizer,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    # --- 6. Тренер ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        tokenizer=tokenizer,
    )

    print("\n=== Начало обучения ===")
    trainer.train()

    # --- 7. Сохранение адаптера ---
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n✓ LoRA-адаптер сохранён в {args.output}")
    print("Для инференса используйте finetune/infer.py")


if __name__ == "__main__":
    main()
