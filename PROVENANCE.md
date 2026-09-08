# Происхождение и изменения

Upstream: https://github.com/aws-samples/sample-ai-possibilities

Исходный checkout: `09a3ebf75bac5014c26326d6de869b8ce7c72169`.
Лицензия upstream сохранена без изменений в LICENSE. Права исходных авторов сохраняются; этот репозиторий не заявляет авторство на AWS samples.

Пользовательская ревизия: AgentF / V6 Direct Attack, подготовленная в ходе проекта NursultanNurmukhamet с помощью AI-инструментов. Изменения относительно samples: Nova Micro model-first policy, ограниченные таймауты, ownership validation, stamina-aware fallback, координация прессинга, anti-corner, own-half clearance, crowded pass, first-touch shots и регрессионные сценарии.

Источник экспорта: финальный `v6-direct-attack-r5-deployed.zip`, SHA-256 `1D6C20CC67D4454BB9AC7B54F4613E0D2DC1E68F72373EF7DC0B4BE8F26C2D24`. Оригинал остался в исходной рабочей папке. Тактические исходники взяты из соответствующей чистой artifact-папки. Экспорт отличается только документацией, запуском/упаковкой, конфигурацией модели/региона через environment и отсутствием исторических targets/скриптов deployment.

Исключены: AWS ARN и account IDs, CloudShell logs, sessions, credentials, старая Git history, node_modules, caches, дубли lib/lib, содержимое корпоративных сервисов, сторонние команды и промежуточные архивы. Старые deploy-скрипты не перенесены: в них присутствовали исторические имена и неодинаковые варианты создания стеков. Вместо них сохранена локальная подготовка независимых runtime-каталогов.

Версии зависимостей закреплены на доступной локальной среде тестирования; полный транзитивный lock исходного AWS build не сохранился. Побитовая воспроизводимость прежнего облачного окружения не заявляется.
