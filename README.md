# Русскоязычная языковая модель на архитектуре Qwen3.5

Проект обучает с нуля текстовый decoder Qwen3.5 через официальную реализацию Hugging Face Transformers. Используется гибридный стек Qwen3.5: три слоя Gated DeltaNet с линейным вниманием на каждый четвёртый слой с полным causal attention, gated MLP, RoPE и общие входные/выходные embeddings. Это конфигурация около 500 млн параметров, созданная по мотивам текстовой конфигурации Qwen3.5-0.8B; готовые веса Qwen не загружаются.

Русский токенизатор `ai-forever/ruGPT-3.5-13B`, словарь из 50 257 токенов и уже подготовленный корпус остаются прежними. Меняется только архитектура модели и каталог запуска. Подготовленные `train.bin`, `val.bin` и токенизатор повторно использовать можно.

## Конфигурация

| Параметр | Значение |
|---|---:|
| Параметры модели | около 500 млн (точное число пишется в начале лога) |
| Слои / скрытый размер | 20 / 1024 |
| Типы слоёв | 15 Gated DeltaNet + 5 full attention |
| Attention / FFN | GQA 8/2; gated MLP, 3584 |
| Словарь / контекст обучения | 50 257 / 1024 токена |
| Обучающие токены | 10 млрд, около 20 токенов на параметр |
| Валидация | 20 млн токенов из отложенных документов |
| Global batch | 262 144 токена |
| Microbatch | 2 последовательности на GPU |
| AdamW | LR 0.0003; betas 0.9, 0.95; eps 1e-8; weight decay 0.1 |
| Расписание LR | Разогрев 1%, cosine decay до 0.00003 |
| Точность | BF16 autocast; состояния AdamW FP32 |

План — примерно 38 147 шагов оптимизатора. На двух GPU каждая карта хранит копию модели; 48 ГБ суммарной памяти не объединяются в один пул. Начните с короткого запуска на 10 шагов, чтобы проверить доступную память конкретных карт.

## Источники данных

Для подготовки используется потоковая смесь из закреплённых версий четырёх русскоязычных наборов:

| Источник | Конфигурация | Вероятность выбора документа |
|---|---|---:|
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | `rus_Cyrl` | 75% |
| [mC4](https://huggingface.co/datasets/allenai/c4) | `ru` | 15% |
| [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) | `ru` | 8% |
| [Taiga](https://huggingface.co/datasets/0x7o/taiga) | `train` | 2% |

Вероятности относятся к документам, а не к токенам: средняя длина документов у источников разная. Фактическая смесь записывается в `manifest.json` в `source_mix`. Если конечный источник исчерпается, подготовка продолжит читать оставшиеся источники без повторного чтения исчерпанного набора. Точные дубликаты документов отбрасываются между всеми источниками.

Валидационные документы отделяются по хешу текста, до токенизации. Из mC4 используется только split `train`; его собственный `validation` не попадает в обучение. FineWeb-2 распространяется по ODC-By; FineWiki — по CC BY-SA 4.0; mC4 — по ODC-By. Официальная страница Taiga ограничивает использование личными и исследовательскими целями, поэтому этот запуск рассчитан на исследовательский эксперимент, не на коммерческое применение.

## Установка

Запускайте на Linux/WSL2 с NVIDIA GPU и доступом к Hugging Face:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('GPU:', torch.cuda.device_count()); print('BF16:', torch.cuda.is_bf16_supported())"
```

Если сервер использует другую совместимую версию CUDA, установите подходящую сборку PyTorch по [официальной инструкции](https://pytorch.org/get-started/previous-versions/). Отдельный CUDA Toolkit для готового wheel PyTorch не требуется.

Для быстрых CUDA-ядер Gated DeltaNet установите совместимые с установленным PyTorch/CUDA пакеты:

```bash
python -m pip install causal-conv1d flash-linear-attention
```

Без них Transformers использует более медленную реализацию на PyTorch.

## Подготовка и запуск

Проверьте конфигурацию и подготовьте смешанный токенизированный корпус. Файлы токенов займут около 20 ГБ; во время подготовки дополнительно нужны место для кеша Hugging Face и временной SQLite-базы удаления дубликатов.

```bash
python prepare_data.py --config configs/adamw_ru_500m.json --output data/russian_mix_10b
```

Короткий запуск на двух GPU. Для новой архитектуры используй новый каталог чекпойнтов; старые чекпойнты от предыдущего Transformer несовместимы:

```bash
python run_experiment.py --gpus 2 --config configs/adamw_ru_500m.json --data data/russian_mix_10b --output runs/qwen35_ru_500m_seed42 --stop-after-steps 10
```

Если в логах нет ошибки по памяти, продолжите до конца бюджета:

```bash
python run_experiment.py --gpus 2 --config configs/adamw_ru_500m.json --data data/russian_mix_10b --output runs/qwen35_ru_500m_seed42 --resume
```

Для запуска одной картой укажите `--gpus 1`. При нехватке памяти уменьшите `micro_batch_size` в конфиге до 1; число шагов накопления градиента увеличится, а размер global batch останется тем же. Для продолжения аварийно прерванного процесса используйте `--resume` с теми же конфигурацией, данными и числом GPU.

## Поэтапное обучение: 1 млрд, затем до 10 млрд

Конфиг рассчитан на корпус в 10 млрд токенов, но первый этап останавливается на 1 млрд. Если `data/russian_mix_10b/manifest.json` уже есть, корпус и токенизатор готовы — команду `prepare_data.py` пропусти:

```bash
python prepare_data.py --config configs/adamw_ru_500m_1b_on_10b.json --output data/russian_mix_10b
python run_experiment.py --gpus 2 --config configs/adamw_ru_500m_1b_on_10b.json --data data/russian_mix_10b --output runs/qwen35_ru_500m_seed42 --stop-after-steps 10
```

Продолжи первый этап до 1 млрд:

```bash
python run_experiment.py --gpus 2 --config configs/adamw_ru_500m_1b_on_10b.json --data data/russian_mix_10b --output runs/qwen35_ru_500m_seed42 --resume
```

После оценки чекпойнта продолжи тот же запуск до 10 млрд всего:

```bash
python run_experiment.py --gpus 2 --config configs/adamw_ru_500m_10b_continue.json --data data/russian_mix_10b --output runs/qwen35_ru_500m_seed42 --resume --continue-training
```

Второй этап использует тот же корпус, манифест и валидацию, сохраняет состояние AdamW и продолжает с постоянным learning rate `0.00003` — конечным уровнем cosine schedule первого этапа. Он продолжает **до 10 млрд всего**, а не добавляет ещё 10 млрд. Каталог `runs/qwen35_ru_500m_seed42` с `latest.pt` и `data_manifest.json` должен сохраниться.

## Результаты

Каталог запуска хранит `config.json`, `data_manifest.json`, копию токенизатора, `metrics.jsonl`, `latest.pt` и лучший чекпойнт `best.pt`. Лог включает train loss, validation loss, perplexity, learning rate, скорость и пиковую память. После обучения оцените лучший чекпойнт на всех 20 млн валидационных токенов:

```bash
python train.py --config configs/adamw_ru_500m.json --data data/russian_mix_10b --output runs/qwen35_ru_500m_seed42 --resume runs/qwen35_ru_500m_seed42/best.pt --eval-only --full-validation
```

Сгенерировать продолжение текста:

```bash
python generate.py --checkpoint runs/qwen35_ru_500m_seed42/best.pt --prompt "В глубине старого леса" --max-new-tokens 100
```

Это базовая языковая модель продолжения текста, не диалоговая модель и не инструкция-следующая модель. Perplexity сравнивайте только при одинаковом токенизаторе и одинаковой отложенной выборке.
