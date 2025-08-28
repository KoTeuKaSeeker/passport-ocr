// static/scripts/main.js
(() => {
    'use strict';

    // ---------- Константы ----------
    const MAX_SIZE = 20 * 1024 * 1024; // 20 MB
    const MIN_SHORT_SIDE = 1000; // рекомендация: короткая сторона >= 1000px
    const ALLOWED_MIMES = ['image/jpeg', 'image/png']; // heic проверяем по расширению

    // ---------- Элементы DOM ----------
    const uploadDrop = document.querySelector('.upload-drop');
    const fileInput = document.querySelector('#file');
    const previewBox = document.querySelector('.preview-box');
    const previewPlaceholder = document.querySelector('.preview-placeholder');
    const docSelect = document.querySelector('#doc-select');
    const form = document.querySelector('.passport-form');
    const submitBtn = form ? form.querySelector('button[type="submit"]') : null;
    const processingHint = document.querySelector('.processing-hint');

    // Новые элементы (из HTML)
    const fillBtn = document.getElementById('fill-btn');
    const fillLoading = document.getElementById('fill-loading');
    const fillControlsContainer = document.querySelector('.fill-controls');

    // Вкладки
    const tabBtnPassport = document.getElementById('tab-btn-passport');
    const tabBtnForeign = document.getElementById('tab-btn-foreign');
    const tabPanelPassport = document.getElementById('tab-panel-passport');
    const tabPanelForeign = document.getElementById('tab-panel-foreign');

    let current_file = null;

    // Сообщения под загрузчиком
    let messagesEl = document.querySelector('.upload-messages');
    if (!messagesEl && uploadDrop) {
        messagesEl = document.createElement('div');
        messagesEl.className = 'upload-messages';
        messagesEl.style.marginTop = '8px';
        messagesEl.style.fontSize = '13px';

        fillControlsContainer.after(messagesEl);
    }

    // ---------- Утилиты ----------
    function showMessage(text, type = 'info') {
        if (!messagesEl) return;
        messagesEl.textContent = text;
        messagesEl.style.color = type === 'error' ? '#b91c1c' : '#374151';
    }

    function clearMessage() {
        if (!messagesEl) return;
        messagesEl.textContent = '';
    }

    function humanFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function isHeicByName(name) {
        return /\.(heic|heif)$/i.test(name || '');
    }

    function isAllowedFile(file) {
        if (!file) return false;
        if (ALLOWED_MIMES.includes(file.type)) return true;
        if (isHeicByName(file.name)) return true;
        return false;
    }

    function setSubmitEnabled(enabled) {
        if (!submitBtn) return;
        submitBtn.disabled = !enabled;
        if (enabled) submitBtn.classList.remove('disabled');
        else submitBtn.classList.add('disabled');
    }

    // Простая защита от XSS при вставке имени файла
    function escapeHtml(str) {
        return String(str).replace(/[&<>"']/g, s =>
            ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[s]
        );
    }

    // ---------- Превью / обработка файла ----------
    function clearPreview() {
        if (!previewBox) return;
        previewBox.innerHTML = '';
        if (previewPlaceholder) previewBox.appendChild(previewPlaceholder);
        if (previewPlaceholder) previewPlaceholder.style.display = '';
        setSubmitEnabled(false);
        clearMessage();
    }

    function renderPreviewInfo(name, size, width, height, warnShortSide) {
        const info = document.createElement('div');
        info.className = 'preview-info';
        info.style.marginTop = '8px';
        info.style.fontSize = '13px';
        info.style.color = '#374151';
        info.innerHTML = `
      <div><strong>${escapeHtml(name)}</strong> — ${humanFileSize(size)}</div>
      <div>Разрешение: ${width} × ${height}</div>
      ${warnShortSide ? `<div style="color:#b45309">⚠️ Короткая сторона меньше ${MIN_SHORT_SIDE}px — качество распознавания может быть снижено.</div>` : ''}
      ${isHeicByName(name) ? `<div style="color:#6b7280">ℹ️ HEIC: будет конвертирован на сервере (если требуется).</div>` : ''}
    `;
        return info;
    }

    async function handleFile(file) {
        clearMessage();
        if (!file) {
            showMessage('Файл не выбран', 'error');
            clearPreview();
            return;
        }

        if (!isAllowedFile(file)) {
            showMessage('Неподдерживаемый формат. Допускаются JPG, PNG, HEIC.', 'error');
            clearPreview();
            return;
        }

        if (file.size > MAX_SIZE) {
            showMessage(`Файл слишком большой: ${humanFileSize(file.size)}. Максимум ${humanFileSize(MAX_SIZE)}.`, 'error');
            clearPreview();
            return;
        }

        current_file = file;

        const objectUrl = URL.createObjectURL(file);
        const img = new Image();
        img.src = objectUrl;

        img.onload = () => {
            const width = img.naturalWidth;
            const height = img.naturalHeight;
            const shortSide = Math.min(width, height);
            const warnShortSide = shortSide < MIN_SHORT_SIDE;

            if (!previewBox) return;
            previewBox.innerHTML = '';
            img.style.maxWidth = '100%';
            img.style.maxHeight = '100%';
            img.alt = 'Превью загруженного документа';
            previewBox.appendChild(img);

            const info = renderPreviewInfo(file.name, file.size, width, height, warnShortSide);
            previewBox.appendChild(info);

            setSubmitEnabled(true);

            if (processingHint) {
                processingHint.textContent = 'После распознавания исходные изображения будут удалены через 3 минуты.';
            }

            setTimeout(() => URL.revokeObjectURL(objectUrl), 30000);
        };

        img.onerror = () => {
            showMessage('Не удалось прочитать изображение. Возможно файл повреждён или неверный формат.', 'error');
            URL.revokeObjectURL(objectUrl);
            clearPreview();
        };
    }

    // ---------- Drag & Drop ----------
    if (uploadDrop) {
        ['dragenter', 'dragover'].forEach(evt =>
            uploadDrop.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadDrop.classList.add('dragover');
            })
        );

        ['dragleave', 'dragend'].forEach(evt =>
            uploadDrop.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                setTimeout(() => uploadDrop.classList.remove('dragover'), 50);
            })
        );

        uploadDrop.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadDrop.classList.remove('dragover');
            const dt = e.dataTransfer;
            if (!dt) return;
            const files = dt.files;
            if (!files || files.length === 0) {
                showMessage('Перетащите файл сюда.', 'error');
                return;
            }
            const file = files[0];
            if (fileInput) {
                try {
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    fileInput.files = dataTransfer.files;
                } catch (err) {
                    // ignore
                }
            }
            handleFile(file);
        });
    }

    // Отключение перетаскивания файла на окно
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
        window.addEventListener(name, (e) => {
            if (!uploadDrop || !uploadDrop.contains(e.target)) {
                e.preventDefault();
            }
        }, false);
    });

    // Обработка выбора файла через input
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const f = e.target.files && e.target.files[0];
            handleFile(f);
        });
    }

    // ---------- Форма: submit & reset ----------
    if (form) {
        form.addEventListener('reset', () => {
            if (fileInput) {
                try { fileInput.value = ''; } catch (e) { }
            }
            // clearPreview();
            // убрать пометку "заполнено" на вкладках
            if (tabBtnPassport) tabBtnPassport.classList.remove('tab--filled');
            if (tabBtnForeign) tabBtnForeign.classList.remove('tab--filled');
            // вернуть вкладку по умолчанию (паспорт)
            activateTab('passport');
            clearMessage();
        });
    }

    // Инициализация: отключаем кнопку подтверждения по умолчанию
    setSubmitEnabled(false);

    // ---------- Tabs: переключение и подсветка ----------
    function activateTab(name) {
        if (!tabBtnPassport || !tabBtnForeign || !tabPanelPassport || !tabPanelForeign) return;
        if (name === 'passport') {
            tabBtnPassport.classList.add('tab--active');
            tabBtnPassport.setAttribute('aria-selected', 'true');
            tabBtnForeign.classList.remove('tab--active');
            tabBtnForeign.setAttribute('aria-selected', 'false');

            tabPanelPassport.classList.add('tab-panel--active');
            tabPanelPassport.removeAttribute('hidden');
            tabPanelPassport.setAttribute('aria-hidden', 'false');

            tabPanelForeign.classList.remove('tab-panel--active');
            tabPanelForeign.setAttribute('hidden', '');
            tabPanelForeign.setAttribute('aria-hidden', 'true');
        } else if (name === 'foreign') {
            tabBtnForeign.classList.add('tab--active');
            tabBtnForeign.setAttribute('aria-selected', 'true');
            tabBtnPassport.classList.remove('tab--active');
            tabBtnPassport.setAttribute('aria-selected', 'false');

            tabPanelForeign.classList.add('tab-panel--active');
            tabPanelForeign.removeAttribute('hidden');
            tabPanelForeign.setAttribute('aria-hidden', 'false');

            tabPanelPassport.classList.remove('tab-panel--active');
            tabPanelPassport.setAttribute('hidden', '');
            tabPanelPassport.setAttribute('aria-hidden', 'true');
        }
    }

    // Навешиваем обработчики на кнопки вкладок (если присутствуют)
    if (tabBtnPassport) {
        tabBtnPassport.addEventListener('click', () => activateTab('passport'));
    }
    if (tabBtnForeign) {
        tabBtnForeign.addEventListener('click', () => activateTab('foreign'));
    }

    // Установим вкладку по умолчанию (passport)
    activateTab('passport');

    // ---------- Предопределённые демонстрационные значения ----------
    const demoValues = {
        // Общие данные
        last_name: 'ИВАНОВ',
        first_name: 'ИВАН',
        middle_name: 'ИВАНОВИЧ',
        gender: 'M',
        birth_date: '01.01.1990',
        birth_place: 'г. Москва',

        // Внутренний паспорт
        series: '1234',
        number: '123456',
        issue_date: '15.06.2015',
        dept_code: '770-001',
        issued_by: 'Отдел УФМС по г. Москве',
        citizenship: 'Российская Федерация',

        // Загран
        latin_name: 'IVANOV IVAN',
        passport_number: '45AB123456',
        issue_date_foreign: '01.01.2018',
        expiry_date: '01.01.2028',
        issued_by_foreign: 'Консульство г. Москвы',
        mrz: 'P<RUSIVANOV<<IVAN<<<<<<<<<<<<<<<<<<<<\n1234567890RUS9001012M2801012<<<<<<<<<<<<<<06',
        citizenship_foreign: 'RUS'
    };

    // ---------- Вспомогательные функции работы с формой ----------
    // Безопасный селектор поиска по имени (использует querySelector)
    function setFieldValue(name, value) {
        if (!form) return false;
        // Экранируем имя поля для селектора (на современных браузерах доступна CSS.escape; если нет — простой fallback)
        const escName = (window.CSS && CSS.escape) ? CSS.escape(name) : name.replace(/([ "'.:\\/[\]#])/g, '\\$1');
        const el = form.querySelector(`[name="${escName}"]`);
        if (!el) return false;

        const tag = el.tagName.toLowerCase();
        if (tag === 'select') {
            // если опция отсутствует — добавляем временную опцию
            let opt = Array.from(el.options).find(o => o.value === value);
            if (!opt) {
                opt = document.createElement('option');
                opt.value = value;
                opt.text = value;
                el.appendChild(opt);
            }
            el.value = value;
            el.dispatchEvent(new Event('change', { bubbles: true }));
        } else if (tag === 'textarea' || (tag === 'input' && (el.type === 'text' || el.type === 'tel' || el.type === 'email' || el.type === 'search'))) {
            el.value = value;
            el.dispatchEvent(new Event('input', { bubbles: true }));
        } else if (tag === 'input') {
            try { el.value = value; } catch (e) { }
        } else {
            try { el.value = value; } catch (e) { }
        }
        return true;
    }

    // Получить первый фокусируемый элемент в панели (для UX)
    function focusFirstFieldInPanel(panelEl) {
        if (!panelEl) return;
        const first = panelEl.querySelector('input, select, textarea, button');
        if (first && typeof first.focus === 'function') first.focus();
    }

    // Пометить вкладку как заполненную (визуальная подсветка)
    function markTabFilled(name) {
        if (!tabBtnPassport || !tabBtnForeign) return;
        if (name === 'passport') {
            tabBtnPassport.classList.add('tab--filled');
            tabBtnForeign.classList.remove('tab--filled');
        } else if (name === 'foreign') {
            tabBtnForeign.classList.add('tab--filled');
            tabBtnPassport.classList.remove('tab--filled');
        }
    }

    async function recognizePassportImage() {
        if (!current_file) {
            console.error("Файл для распознавания не выбран");
            return null;
        }

        const formData = new FormData();
        formData.append("file", current_file, current_file.name);

        try {
            const res = await fetch("/api/recognize-passport", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error("Ошибка запроса: " + res.status);
            }

            const data = await res.json();
            console.log("Результат распознавания:", data);
            return data;
        } catch (err) {
            console.error("Ошибка при отправке:", err);
            return null;
        }
    }


    // ---------- Поведение кнопки "Заполнить форму" (рандомный выбор вкладки) ----------
    let fillInProgress = false;

    if (fillBtn) {
        fillBtn.addEventListener('click', (e) => {
            if (fillInProgress) return;
            fillInProgress = true;

            // Показать индикатор загрузки
            if (fillLoading) {
                fillLoading.hidden = false;
                fillLoading.setAttribute('aria-busy', 'true');
            }
            fillBtn.disabled = true;
            fillBtn.setAttribute('aria-disabled', 'true');

            // Сообщение (если превью пустой — предупреждаем)
            if (previewPlaceholder && previewBox.contains(previewPlaceholder)) {
                showMessage('Демо: заполняем форму без загруженного изображения (примерные данные).', 'info');
            } else {
                clearMessage();
            }

            // Эмулируем время распознавания
            const delay = 800 + Math.floor(Math.random() * 1200);

            setTimeout(() => {

                recognizePassportImage().then(data => {
                    console.log("Распознавание завершено:", data);

                    // случайным образом выбираем вкладку: passport или foreign
                    const chosen = data.passport_type === 'foreign' ? 'foreign' : 'passport';

                    // Открываем выбранную вкладку
                    activateTab(chosen);

                    for (const key in data.fields) {
                        if (data.fields.hasOwnProperty(key)) {
                            setFieldValue(key, data.fields[key]);
                        }
                    }

                    // Пометим вкладку как заполненную и выделим её визуально
                    markTabFilled(chosen);

                    // скрыть индикатор, разблокировать кнопку
                    if (fillLoading) {
                        fillLoading.hidden = true;
                        fillLoading.removeAttribute('aria-busy');
                    }
                    fillBtn.disabled = false;
                    fillBtn.removeAttribute('aria-disabled');
                    fillInProgress = false;

                    // UX: фокус на первом поле выбранной панели
                    const panel = chosen === 'passport' ? tabPanelPassport : tabPanelForeign;
                    focusFirstFieldInPanel(panel);

                    showMessage('Форма успешно заполнена (демонстрационные значения).', 'info');

                }).catch(err => {
                    console.error("Ошибка распознавания:", err);
                    showMessage('Ошибка распознавания изображения.', 'error');
                    return;
                }).finally(() => {
                    // скрыть индикатор, разблокировать кнопку
                    if (fillLoading) {
                        fillLoading.hidden = true;
                        fillLoading.removeAttribute('aria-busy');
                    }
                    fillBtn.disabled = false;
                    fillBtn.removeAttribute('aria-disabled');
                    fillInProgress = false;
                });
            }, delay);
        });
    }

    // ---------- Инициализация состояния ----------
    // Если нет загруженного превью — явно выставим плейсхолдер
    if (previewBox && previewBox.querySelector('img') == null) {
        clearPreview();
    }

    // Возврат вкладки к дефолту при загрузке (уже вызван выше, но на всякий)
    activateTab('passport');

})();
