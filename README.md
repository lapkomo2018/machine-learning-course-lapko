# Основи машинного навчання

Навчальний репозиторій для практичних і лабораторних робіт із дисципліни «Основи машинного навчання».

## Виконані роботи

### Практична робота №1

[Аналіз і підготовка Bike Sharing Dataset](practical01/README.md), варіант 7.

- Проведено аудит 17 379 погодинних спостережень і 17 стовпців.
- Пропусків і повних дублікатів не виявлено.
- `casual` і `registered` вилучено як прямий витік цільової змінної `cnt`.
- Досліджено розподіл попиту, добовий профіль, погоду, температуру та кореляції.
- Сформульовано п’ять гіпотез для подальшого моделювання.
- Побудовано preprocessing pipeline, який після хронологічного поділу формує 60 ознак без пропусків.

Повний аналіз міститься у [виконаному ноутбуці](practical01/notebook/practical01.ipynb).

## Дані

Використано погодинний [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) із UCI Machine Learning Repository. Цільова змінна `cnt` — сумарна кількість оренд протягом години.

## Структура

```text
.
├── data/                  # набір даних і опис джерела
├── src/                   # спільний preprocessing-код
├── practical01/           # практична робота №1
├── requirements.txt
└── README.md
```

## Відтворення практичної роботи

Потрібен Python 3.12 або новіший.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook practical01/notebook/practical01.ipynb
```

Випадкові процедури використовують `random_state=42`.

