# Дані

Файл `hour.csv` — погодинні дані про оренду велосипедів системи Capital Bikeshare за 2011–2012 роки.

- Джерело: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
- Офіційний архів: <https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip>.
- Ліцензія: Creative Commons Attribution 4.0 International (CC BY 4.0), як указано на сторінці набору.
- Розмір `hour.csv`: 17 379 рядків і 17 стовпців.
- Цільова змінна: `cnt` — сумарна кількість погодинних оренд.

`casual` і `registered` не використовуються як вхідні ознаки, бо `cnt = casual + registered`. Їх використання означало б прямий витік цільової змінної. `instant` є технічним індексом, а `dteday` дублює часові ознаки `yr`, `mnth`, `weekday` та `hr`, тому ці два поля також вилучаються перед моделюванням.

