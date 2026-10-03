# AI_USAGE — Практична робота №2

Під час роботи використовувався генеративний ШІ для написання коду notebook, побудови графіків і чернетки тексту. Нижче — суттєві випадки, де пропозиція ШІ вплинула на рішення.

## AI interaction 1 — схема крос-валідації

**Problem:** вибрати схему 5-fold CV для порівняння моделей.
**AI suggestion:** використати `KFold(n_splits=5, shuffle=True, random_state=42)`, як у прикладі методички (так був написаний і ранній чорновий `main.py`).
**Decision:** відхилено. Дані погодинні й часові, тому використано `TimeSeriesSplit(n_splits=5)`.
**Verification:** RF оцінено обома схемами на тому самому train: shuffled KFold — RMSE 51,32, TimeSeriesSplit — 94,53; хронологічний test дав 91,64, тобто правдивою виявилась саме часова оцінка (`results/leakage_checks.csv`).

## AI interaction 2 — поділ train/test

**Problem:** сформувати test set.
**AI suggestion:** `train_test_split(test_size=0.2, random_state=42)`.
**Decision:** відхилено, використано хронологічний поділ 80/20 з `src/bike_workflow.py`.
**Verification:** випадковий split дає test RMSE 51,07 проти 91,64 на хронологічному — оцінка завищується майже вдвічі.

## AI interaction 3 — регуляризація Ridge

**Problem:** чи включати Ridge як окремого кандидата.
**AI suggestion:** Ridge має стабілізувати коефіцієнти через сильну кореляцію `temp`/`atemp` і покращити RMSE.
**Decision:** змінено — Ridge залишено, але як перевірку гіпотези, а не очікуване покращення.
**Verification:** CV RMSE Ridge 108,83 проти 108,76 у LinearRegression; per-fold значення відрізняються в межах 0,2. Твердження про покращення не підтвердилось.

## AI interaction 4 — стандартне відхилення

**Problem:** як рахувати розкид RMSE між fold-ами.
**AI suggestion:** `scores.std()` (NumPy, `ddof=0`).
**Decision:** змінено на `ddof=1`, бо методичка задає вибіркове std з `k−1`.
**Verification:** для RF std = 23,36 (`ddof=1`) замість ≈ 20,9 (`ddof=0`); у README зазначено, чому значення відрізняються від ЛР1.
