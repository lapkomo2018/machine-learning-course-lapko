# Дані

## Bike Sharing (`hour.csv`) — ПР1, ЛР1, ПР2, ПР3

Файл `hour.csv` — погодинні дані про оренду велосипедів системи Capital Bikeshare за 2011–2012 роки.

- Джерело: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
- Офіційний архів: <https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip>.
- Ліцензія: Creative Commons Attribution 4.0 International (CC BY 4.0), як указано на сторінці набору.
- Розмір `hour.csv`: 17 379 рядків і 17 стовпців.
- Цільова змінна: `cnt` — сумарна кількість погодинних оренд.

`casual` і `registered` не використовуються як вхідні ознаки, бо `cnt = casual + registered`. Їх використання означало б прямий витік цільової змінної. `instant` є технічним індексом, а `dteday` дублює часові ознаки `yr`, `mnth`, `weekday` та `hr`, тому ці два поля також вилучаються перед моделюванням.

## Titanic (`titanic.csv`) — ЛР2, ЛР3, ПР4, ЛР4

Варіант 7 класифікаційних робіт.

- Джерело: [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data), файл `train.csv` (891 пасажир, 12 стовпців).
- Копія без змін для відтворюваності завантажена з <https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv>.
- Ціль: `Survived`; у роботах позитивний клас `AtRisk = 1 − Survived` (пасажир не вижив).
- `PassengerId`, `Name`, `Ticket` — ідентифікатори, `Cabin` має 77% пропусків; ці стовпці не використовуються як ознаки.
- Персональні дані — історичні публічні записи, поширені в навчальних цілях.
