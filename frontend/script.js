document.getElementById('file-input').addEventListener('change', async function () {
    const file = this.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/api/predict_weather", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            alert("Ошибка при загрузке файла: " + response.status);
            return;
        }

        const result = await response.json();
        displayResultAsTable(result);
    } catch (error) {
        alert("Ошибка запроса: " + error);
    }
});

function displayResultAsTable(data) {
    let parsedData;
    try {
        parsedData = typeof data === 'string' ? JSON.parse(data) : data;
    } catch (e) {
        console.error("Ошибка при разборе данных:", e);
        alert("Ошибка: не удалось разобрать данные.");
        return;
    }

    if (!Array.isArray(parsedData)) {
        console.error("Ожидался массив, но получено:", parsedData);
        alert("Ошибка: данные не являются массивом.");
        return;
    }

    const tableContainer = document.getElementById('tableContainer');
    tableContainer.innerHTML = '';

    const table = document.createElement('table');

    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');

    const indexHeader = document.createElement('th');
    indexHeader.textContent = '№';

    const dateHeader = document.createElement('th');
    dateHeader.textContent = 'Дата';

    const classHeader = document.createElement('th');
    classHeader.textContent = 'Класс';

    headerRow.appendChild(indexHeader);
    headerRow.appendChild(dateHeader);
    headerRow.appendChild(classHeader);
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');

    // Сопоставление классов и иконок
    const iconMap = {
        "Солнечно": "clear-day.svg",
        "Малая Облачность": "cloudy-1-day.svg",
        "Переменная Облачность": "cloudy-3-day.svg",
        "Облачно": "cloudy.svg",
        "Морось": "hail.svg",
        "Дождь": "rainy.svg",
        "Снег": "snowy.svg"
    };

    parsedData.forEach((item, index) => {
        const row = document.createElement('tr');

        const indexCell = document.createElement('td');
        indexCell.textContent = index + 1;

        const dateCell = document.createElement('td');
        dateCell.textContent = new Date(item.date).toLocaleString('ru-RU', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
        });

        const classCell = document.createElement('td');
        // Создание содержимого ячейки с иконкой
        const iconSrc = iconMap[item.class];
        if (iconSrc) {
            const iconImg = document.createElement('img');
            iconImg.src = `weather_icons/${iconSrc}`;
            iconImg.alt = item.class;
            iconImg.classList.add('weather-icon');
            classCell.appendChild(iconImg);
        }

        const textNode = document.createTextNode(item.class);
        classCell.appendChild(textNode);

        row.appendChild(indexCell);
        row.appendChild(dateCell);
        row.appendChild(classCell);
        tbody.appendChild(row);
    });

    table.appendChild(tbody);

    const downloadButton = document.createElement('button');
    downloadButton.textContent = 'Скачать данные';
    downloadButton.onclick = function() {
        downloadDataAsFile(parsedData);
    };
    tableContainer.appendChild(downloadButton);
    tableContainer.appendChild(table);
}