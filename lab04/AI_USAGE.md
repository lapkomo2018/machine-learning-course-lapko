# AI_USAGE — Лабораторна робота №4

Генеративний ШІ використовувався для коду notebook (nested CV, bootstrap, permutation importance, вимірювання ресурсів), графіків і чернетки звіту. Обов'язковий критичний аудит — у `README.md`, розділ «Аудит рекомендації ШІ».

## Task 1 — інтерпретація `feature_importances_`

**Proposal:** ознака з найбільшою impurity importance — головна причина класу; для RF другою названо `Fare`.
**Assumptions:** impurity не зміщена; ознаки незалежні; поведінка моделі = причинний механізм.
**Decision:** змінено.
**Verification:** permutation importance на валідаційних fold-ах (`results/permutation_importance_folds.csv`): `Fare` 0,010 ± 0,023 при impurity 0,191; Spearman `Fare–Pclass` = −0,70; корінь усіх 150 bootstrap-дерев — `Sex_female`. Висновок сформульовано як опис поведінки моделі без причинних тверджень.

## Task 2 — «ансамбль стабільніший за дерево»

**Proposal:** випадковий ліс завжди дає стабільніші прогнози, ніж окреме дерево.
**Decision:** змінено після експерименту.
**Verification:** `results/stability_summary.csv`: розбіжність з консенсусом за 10 seed-ами — необмежене дерево 8,90%, RF 3,57%, HGB 4,47%, дерево глибини 3 — 2,23%. Твердження правильне лише для глибоких дерев.

## Task 3 — вибір структури дерева

**Proposal:** обирати `max_depth`, `min_samples_leaf` за максимумом CV у `GridSearchCV`.
**Decision:** відхилено після першого запуску (60 листків, перевага 0,012 < SE 0,018); застосовано правило однієї SE через `refit`-функцію і в основному пошуку, і всередині nested CV. Зміну зафіксовано в README до відкриття test.
**Verification:** `results/tree_structure_search.csv` (повна сітка), `configs/final_model.json`.

## Task 4 — код nested CV

**Proposal:** оцінювати налаштоване дерево через `cross_validate(GridSearchCV(...), cv=outer_cv)`.
**Decision:** прийнято; перевірено, що test не потрапляє в жоден fold (`X_train` лише), а внутрішній `StratifiedKFold(4)` має інший `random_state`, ніж зовнішній.
