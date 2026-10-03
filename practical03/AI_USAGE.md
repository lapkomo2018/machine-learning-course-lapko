# AI_USAGE — Практична робота №3

Генеративний ШІ використовувався для коду notebook, графіків і чернетки звіту. Повна таблиця аудиту тверджень ШІ — у `README.md`, розділ «Аудит рекомендацій ШІ». Ключові випадки:

## AI interaction 1 — поліном по всіх ознаках

**Problem:** як побудувати шкалу складності.
**AI suggestion:** `PolynomialFeatures(degree)` до всіх 10 вхідних ознак, `KFold(shuffle=True)` (ранній чорновий `main.py`).
**Decision:** змінено. Поліном лише по 5 числових ознаках, категоріальні — one-hot без поліному; CV — `TimeSeriesSplit(5)`.
**Verification:** аналітично `C(60+3,3) − 1 = 39 710` стовпців для повного one-hot простору; формулу `C(p+d,d) − 1` перевірено проти `PolynomialFeatures.n_output_features_` (55 = 55 для d=3).

## AI interaction 2 — ефект регуляризації

**Problem:** чи виправдана регуляризована складна модель.
**AI suggestion:** Ridge на поліномі високого степеня дасть кращу модель, ніж лінійна.
**Decision:** змінено — прийнято, що регуляризація усуває перенавчання, відхилено, що вона дає кращу модель.
**Verification:** `results/final_comparison.csv`: Ridge d=5 CV 119,53, Lasso d=5 111,36, OLS d=1 108,76; правило 1-SE обирає OLS d=1.

## AI interaction 3 — інтерпретація Lasso

**Problem:** що означають нульові коефіцієнти.
**AI suggestion:** нульові коефіцієнти Lasso — неважливі ознаки.
**Decision:** змінено формулювання.
**Verification:** множини ненульових коефіцієнтів на 5 схемах CV мають середній Jaccard 0,81, кількість 42–58 (`results/stability_summary.csv`).

## AI interaction 4 — крива навчання

**Problem:** побудувати криву навчання для часових даних.
**AI suggestion:** `sklearn.model_selection.learning_curve(..., cv=TimeSeriesSplit(5))`.
**Decision:** відхилено після першого запуску.
**Verification:** максимальний розмір підвибірки дорівнював 2 318 (найменший fold), і бралися перші години 2011 р.; крива перебудована вручну на найближчій історії до трьох останніх fold-ів (`results/learning_curves.csv`).
