# Запуск демонстрации

Для запуска нужны Docker с Compose и папка `demo-assets` рядом с `compose.yaml`. В ней лежат экспортированные модели и исходные примеры для обоих профилей. После завершения итоговой оценки (`scripts/quality_selection.py 6`) команда `python scripts/prepare_quality_demo.py` в окружении обучения создаёт эту папку и проверяет экспорт ONNX.

Из каталога проекта запустите сборку и контейнер:

```powershell
docker compose up --build
```

В PowerShell на компьютере с Docker Engine внутри Ubuntu WSL команда выглядит так:

```powershell
wsl -d Ubuntu --cd "D:\CodesNodes_2\Garbardge detector" -- docker compose up --build
```

Откройте [локальную страницу](http://localhost:7860), выберите «Суша» или «Под водой» и запустите демо-фото, демо-видео либо свой файл. Размеченный результат и JSON доступны для скачивания со страницы. Для остановки нажмите `Ctrl+C` или запустите `docker compose down` в той же среде.

Размер запроса ограничен 128 МиБ. Видео может содержать до 3600 кадров. Сервер обрабатывает по одному файлу за раз.

Если Docker умеет передавать NVIDIA GPU в контейнер, можно запустить профиль ускорения командой `docker compose --profile gpu up --build demo-gpu`.
