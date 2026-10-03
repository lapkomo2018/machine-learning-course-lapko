# Основи машинного навчання

Навчальний репозиторій для практичних і лабораторних робіт із дисципліни «Основи машинного навчання». Варіант 7 виконано на погодинному Bike Sharing Dataset із цільовою змінною `cnt` — кількістю оренд велосипедів.

## Виконані роботи

### Практична робота №1

[Аналіз і підготовка Bike Sharing Dataset](practical01/README.md).

- Проведено аудит 17 379 погодинних спостережень і 17 стовпців.
- Пропусків і повних дублікатів не виявлено.
- `casual` і `registered` вилучено як прямий витік цільової змінної.
- Досліджено розподіл попиту, добовий профіль, погоду, температуру та кореляції.
- Сформульовано п’ять гіпотез і побудовано preprocessing pipeline на 60 вихідних ознак.

Повний аналіз міститься у [виконаному ноутбуці практичної](practical01/notebook/practical01.ipynb).

### Лабораторна робота №1

[Побудова та дослідження регресійних моделей](lab01/README.md).

- Використано хронологічний поділ 80/20 і п’ять часових fold.
- Порівняно baseline, лінійну, поліноміальну Ridge та HistGradientBoosting моделі.
- За CV RMSE обрано `HistGradientBoostingRegressor` із `max_leaf_nodes=15`.
- На незалежному тесті отримано MAE `55,887`, RMSE `79,773` і R² `0,869`.
- Проаналізовано залишки та об’єкти з найбільшими абсолютними похибками.

Повний експеримент міститься у [виконаному ноутбуці лабораторної](lab01/notebook/lab01.ipynb).

### Практична робота №2

[Побудова та оцінювання коректного ML-експерименту](practical02/README.md) — Bike Sharing.

- Хронологічний split 80/20 і `TimeSeriesSplit(5)` для baseline, Linear, Ridge, Decision Tree і Random Forest.
- Обрано Random Forest: CV RMSE 94,53 ± 23,36, test RMSE 91,64 (R² 0,827).
- Показано, що перемішаний KFold і випадковий split занижують RMSE майже вдвічі, а `casual/registered` дають фіктивний RMSE ≈ 0.

### Практична робота №3

[Перенавчання, складність моделі та регуляризація](practical03/README.md) — Bike Sharing.

- Поліноми погоди d = 1…8: з d = 4 CV RMSE «вибухає» через екстраполяцію (159 → 205 895).
- Ridge / Lasso / Elastic Net на d = 5: Lasso (α = 0,1) знижує CV RMSE з 260 до 111.
- За правилом однієї SE обрано OLS d = 1 (CV 108,76, test 133,84).

### Лабораторна робота №2

[Класифікація та аналіз помилок моделі](lab02/README.md) — Titanic, позитивний клас «не вижив».

- Baseline, Logistic Regression і GaussianNB; обрано LR (CV macro F1 0,789, ROC-AUC 0,854).
- Test: Accuracy 0,810, macro F1 0,798; аналіз 18 FP і 16 FN, пороги 0,3/0,5/0,7, рекомендовано t = 0,4.

### Лабораторна робота №3

[Метричні методи та SVM](lab03/README.md) — make_moons і Titanic.

- Масштабування, NearestCentroid, kNN (32 конфігурації), linear і RBF SVM (21), експеримент із розмірністю.
- Обрано RBF SVM (C = 100, γ = 0,01): CV macro F1 0,800, test 0,808; розбіжності kNN/SVM проаналізовано на OOF.

### Практична робота №4

[Пороги, вартість помилок і прийняття рішень](practical04/README.md) — Titanic.

- OOF-бали, калібрування, три сценарії вартості, місткість 50%; робоча точка t = 0,71.
- Політика як компонент: `practical04/src/decision_policy.py`, конфігурація JSON і 18 тестів.

### Лабораторна робота №4

[Дерева рішень та ансамблі](lab04/README.md) — Titanic.

- Nested CV дерева, Random Forest, HistGradientBoosting на 20 спільних fold-ах; стійкість за 10 seed-ами і 150 bootstrap.
- Ансамблі кращі лише на 0,011–0,014 macro F1; обрано дерево глибини 3 (test macro F1 0,800).

## Дані

- ПР1–ПР3, ЛР1: погодинний [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) із UCI Machine Learning Repository. Цільова змінна `cnt` — сумарна кількість оренд протягом години.
- ЛР2–ЛР4, ПР4 (варіант 7 класифікаційних робіт): [Titanic](https://www.kaggle.com/c/titanic/data), ціль `Survived`, позитивний клас `AtRisk = 1 − Survived`.

Деталі — у [data/README.md](data/README.md).

## Структура

```text
.
├── data/                  # набір даних і опис джерела
├── src/                   # спільний відтворюваний код
├── practical01–04/        # практичні роботи №1–4
├── lab01–04/              # лабораторні роботи №1–4
├── requirements.txt
└── README.md
```

## Відтворення результатів

Потрібен Python 3.12 або новіший.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook
```

Після запуску Jupyter можна виконати notebook будь-якої роботи (`<робота>/notebook/<робота>.ipynb`); кожен зберігає свої таблиці в `results/` і рисунки в `figures/`. Тести політики ПР4: `cd practical04 && python -m unittest -v tests/test_decision_policy.py`. Фінальні версії робіт позначено тегами `pr01-final` … `pr04-final`, `lab01-final` … `lab04-final`. Випадкові процедури використовують `random_state=42`. Тестову вибірку (для Bike Sharing — останні 20% годин, для Titanic — стратифіковані 20%) не використано для вибору моделі.

