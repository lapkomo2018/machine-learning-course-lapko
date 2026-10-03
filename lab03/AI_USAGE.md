# AI_USAGE — Лабораторна робота №3

Генеративний ШІ використовувався для коду notebook, графіків і чернетки звіту. Обов'язковий аудит рекомендацій — таблиця в `README.md`, розділ «Аудит рекомендацій ШІ». Окремі випадки:

## AI interaction 1 — автоматичний StandardScaler

**Task:** вибрати preprocessing для kNN і SVM.
**Proposal:** «завжди застосовувати StandardScaler для метричних методів і SVM».
**Decision:** змінено — масштабування перевірено як гіпотезу: фіксовано модель і її параметри, змінювався лише scaler.
**Verification:** `results/scaling_experiment.csv`: kNN 0,676 → 0,753 (Standard) / 0,752 (MinMax), RBF 0,589 → 0,791 / 0,784, linear SVM 0,768 у всіх варіантах.

## AI interaction 2 — зважений kNN

**Task:** налаштувати kNN.
**Proposal:** використовувати `weights="distance"`, бо ближчі сусіди інформативніші.
**Decision:** відхилено за результатами сітки на однакових fold-ах.
**Verification:** найкращий distance-kNN 0,768 проти uniform 0,789 (`results/knn_grid.csv`); train macro F1 distance ≈ 0,98 — у даних 127 дублікатів векторів, сусід на відстані 0 отримує нескінченну вагу.

## AI interaction 3 — пояснення linear SVM

**Task:** пояснити, чому linear SVM нечутлива до `C`.
**Proposal (гіпотеза ШІ):** лінійна межа з hinge-loss зводиться до найсильнішої ознаки — статі.
**Decision:** прийнято після перевірки.
**Verification:** OOF-прогнози linear SVM збігаються з правилом «male → AtRisk» для 100% пасажирів, macro F1 правила = 0,768 (клітинка в notebook, розділ 7).

## AI interaction 4 — вибір фінальної моделі

**Task:** визначити найкращу модель для системи.
**Proposal:** RBF SVM — найкраща модель, бо має найвищий CV macro F1.
**Decision:** змінено — RBF SVM обрано за заздалегідь заданим правилом (менша серед рівноцінних), але прямо вказано, що різниця з kNN менша за 1 SD, а для сценарію з жорсткою затримкою кращою є логістична регресія.
**Verification:** `results/model_comparison.csv` (якість, час, розмір).
