# README #

Данный код дает дает возможность автоматизировать обучение pytorch моделей. Если вкратце, то:

* Создать датасет
* определить архитектуру сети
* нажать на кнопку
* PROFIT

Основные плюшки:

* автоматическое разбиение на обучающий и валидационный наборы
* ранний останов, отбор лучших версий сети
* усреднение лучших версий сети
* возможность передачи весов классов и весов примеров
* автоматический подбор комбинации классификации и регрессии, возможность обучения смешанной задачи
* автоматический подбор и применение коэффициентов шкалирования
* крутой оптимизатор - комбинация RAdam и Lookahead
* обучение с рестартами, подбор скорости обучения
* простое сохранение и загрузка моделей
* простое применение штучно или к датасету целиком

Типичный шаблон использования может выглядеть так:

```
model = PytorchTrainer(net_builder, [], device='cuda')
model.train(dataset, name="model_name", batch_size=16, n_epochs=10)
model.load("model_name")
model.predict_dataset(dataset)
```
#### Инициализация модели ####

`PytorchTrainer(net_builder, net_build_args, device='cuda')`

* net_builder - функция, возвращающая свежую сеть
* net_builder_args - аргументы, передаваемые в вызов net_builder
* device - 'cpu', 'cuda', 'cuda:1' и т.д.

*Комментарии:*
Такой механизм инициализации сделан, чтобы сохранять состояния сети с помощью state_dict(), т.е. только веса.
Это предпочтительный метод сериализации в pytorch, т.к. сериализация модели целиком
привязывается к конкретным путям и названиям файлов в момент сериализации.
Таким образом, при загрузке сохраненной модели будет вызвана функция net_builder с аргументами net_builder_args.
Список аргументов может быть пустой, а сама функция - просто возвращать заранее инициализированный объект.

#### Обучение модели ####
```
self.train (self, dataset, name="pytorch_model",
            batch_size=16, criterion='auto', n_epochs=50, train_prop=0.9,
            learning_rate=1e-3, n_workers=0, verbose=True, epoch_hook=None,
            repeat_train=1, repeat_valid=1, n_best_nets=1, with_restarts=True,
            shuffle=True):
```

* dataset - датасет для обучения, должен наследовать класс torch.utils.data.Dataset.
подробнее о датасетах см. ниже

* name - имя, под которым будет сохраняться результат данного обучения.
Внимание, имя - не атрибут модели, каждое обучение (если их несколько) может иметь свое имя
и, соответственно, свой путь сериализации.

* batch_size=16
* criterion='auto' - критерий обучения. Подробную информацию о критериях см ниже.
* n_epochs=50
* train_prop=0.9 - пропорция, в которой случайно делится переданный датасет,
0.9 означает 90% на обучение, 10% на валидацию
* learning_rate=1e-3
* n_workers=0 - количество дополнительных параллельных процессов для генерации данных из датасета.
* verbose=True - рисовать прогресс-бары и дополнительную информацию об обучении
* epoch_hook=None - None либо функция, принимающая аргумент - саму модель (self),
вызывается после каждой эпохи, например, для визуальной валидации или записи кривой обучения
* repeat_train=1 - сколько раз дублировать обучающий набор в рамках одной эпохи
* repeat_valid=1 - сколько раз дублировать валидационный набор в рамках одной эпохи
Эти параметры могут быть полезны, если в датасете используется онлайн-аугментация,
приводящая к значительному случайному разбросу в генерируемых данных.
Значения лосс-функции будут усреднены по всем прогонам,в результате кривая обучения
меньше страдает от выбросов при аугментации.
* n_best_nets=1 - количество лучших версий сети, отобранных и усредненных по итогам обучения
* with_restarts=True - использовать обучение с рестартами.
При обучении с рестартами рекомендуется ставить n_best_nets=1, иначе усреднение может ухудшить модель.
Впрочем, при n_best_nets > 1 неусредненная сеть так же сохраняется, так что к потере информации это не приведет.
* shuffle=True - перемешивать примеры в датасете до разбиения на train и valid


*Комментарии:*

При обучении используется механизм раннего останова.
При отсутствии прогресса на валидационной выборке запускается обратный отсчет эпох,
после чего обучение останавливается. длина отсчета - 15% от заданного параметра n_epochs.
Счетчик сбрасывается, если был достигнут прогресс в обучении.
Прогрессом считается улучшение не менее чем на 0.5% от ранее зафиксированного минимума.

При отсутствии прогресса в течении нескольких эпох скорость обучения уменьшается по правилу золотого сечения.
После рестарта скорость будет восстановлена до исходной.

Каждый раз при одновлении минимума ошибки на валидационной выборке сеть сохраняется по пути `$(model_name).pt.`
Если выбран n_best_nets > 1, то после обучения также будет сохранена усредненная сеть по пути `$(model_name)_avg.pt`

Сети для усреднения отбираются по минимальной ошибке из всей истории обучения.
При непосредственном усреднении выбираются только те, для которых значение ошибки отличалось от достигнутого минимума
не более, чем на 10%. Таким образом, версии сети, которые могли бы испортить усреднение, отбрасываются.

Если используется with_restarts, то после истечения 20% от лимита раннего останова выполняется рестарт оптимизатора
(исходя из предположения, что обучение скоро застрянет в локальном минимуме).
Следующий рестарт не будет сделан, пока не будет сброшен счетчик раннего останова,
давая возможность перед завершением обучения сойтись хотя бы к какому-то минимуму.

После завершения обучения в памяти остается неопределенная сеть - усредненная в случае использованя усреднения,
и последняя в ином случае. Для дальнейшего использования или дообучения рекомендуется загрузить нужную модель с диска.

Если требуется принудительно завершить обучение модели, не нужно нажимать ctrl-C.
Если создать в корне питоньего процесса файл с именем `stop_$(name)`, где name - имя, переданное при запуске обучения,
то обучение завершится после завершения текущей эпохи штатно (будет создана усредненная модель, напечатан отчет и т.д.).

При разбиении датасета на обучающий и проверочный, даже с shuffle=True разбиение воспроизводится при последующем разбиении
того же сета, т.е. при повторном обучении на датасете индексы разбиения будут те же, и не будет оверфита.

#### Загрузка модели ####

```
@classmethod
PytorchTrainer.load(name, device='cuda')
```
Загржает с диска модель `$(name).pt` и помещает в заданный device.

#### Применение модели ####

`self.predict_dataset (self, dataset, batch_size=1, verbose=True):`

Применение модели на сформированном датасете.  

* dataset - pytorch датасет в том же формате, который использовался для обучения данной модели
* batch_size=1 - размер батча при применении, может не совпадать с размером батча при обучении
* verbose=True - рисовать прогресс-бары и дополнительную информацию

Датасет должен содержать целевые значения.
Если же они неизвестны, то в датасет следует забить файковые целевые значения нужного размера.

`def predict (self, *data):`

Применение к сырым данным дез формаирования датасета.

`data` - list сырых входных данных в том виде, как они подаются в сеть.

#### Датасет ####

Датасет должен наследовать класс torch.utils.data.Dataset, чтобы в процессе обучения из него был создан DataLoader.
Датасет должен иметь методы len() и getitem(), getitem может возвращать два или три элемента. 

Если возвращается два элемента, то они считаются как input и target, вход и цель обучения.

Если возвращается три элемента, то они будут использованы как input, weight, target, где weight - вес данного примера в обучении.

* input - то, что передается в сеть.

Если net.forward принимает один аргумент, то input должен быть тензором.
Если net.forward принимает больше одного аргумента (например, сеть с интегральными параметрами, сиамская сеть и т.д.),
то input должен быть списком массивов, которые передаются в сеть.
Таким образом, если `net.forward(x1, x2)`, то input должен быть `[x1, x2]`.

* target - тензор той же размерности, которую возвращает сеть
* weight - если передано число, то представляет вес данного пример. Если передан вектор размерности выходного слоя -
то представляет веса каждого выхода (класса). Вектор весов не нормируется, поэтому может передавать одновременно
как распределение весов между выходными классами, так и важность данного примера.

Если датасет нетипичен и требует задания собственной collate_fn - функции,
определяющей порядок склейки отдельных примеров в батч, то необходимо определить соответствующий метод collate_fn в датасете.
Этот метод будет передан в DataLoader.

Иногда при обучении требуется присвоение определенной метки отдельному батчу, например, один батч должен содержать
только положительные или же только отрицательные примеры. В таком случае требуется определить в датасете метод switch_state,
который вызывается при старте генерации каждого нового батча, и нужным образом переключает состояние датасета. 

Для работы со стандартным критерием обучения может потребоваться определения в датасете дополнительных параметров -
regression_flags и scale. (см ниже)

#### Критерий обучения ####

Критерий обучения - не стандартная концепция, она была введена для автоматизации применения с логистическими лосс-функциями,
а также автоматического шкалирования выходных данных. Предположительно, в подавляющем количестве задач хватит возможностей
стандартного MixedCriterion, описанного ниже. Однако есть возможность определить собственный критерий.

При инициализации критерий должен получить аргумент - датасет (либо None если анализировать датасет не требуется).
Критерий должен иметь методы call и postproc. Метод call используется для вычисления loss-функции при обучении.

`def __call__ (self, output, target, weights=None, unscaled=False)`

* output - выход сети
* target, weights - данные из датасета
* unscaled - флаг, говорящий, используются ли в данном вызове шкалированные данные. По умолчанию данные шкалируются.

В процессе обучения шкалированная лосс-функция используется для расчета шага обратного распространения,
а нешкалированная - как критерий качества прогноза, который используется для раннего останова, отбора лучших сетей,
выводится в отчете. Если не требуется шкалирование в собственном критерии, данный параметр будет игнорироваться.
(Шкалирование - это приведение вектора к среднему значению 0 и стандартному отклонению 1, имеет смысл только для задачи регрессии.)

`def postproc (self, pred):`

Метод используется в применении модели для компенсации разницы между реальным и требуемым выходом сети,
связанным с несимметричностью лосс-функции. Например, для кросс-энтропийных лосс-функций postproc должен возвращать
torch.sigmoid(pred), где pred - сырой выход сети. Также, если критерий использует шкалирование,
в методе должна выполняться операция unscale. 

#### Mixed Criterion ####

По умолчанию используется класс MixedCriterion, его можно использовать, если задача -
любая комбинация регрессии и бинарной классификации. Пусть требуется прогнозировать две колонки, из них первая -
регрессия, вторая - бинарная классификация. В таком случае к первой колонке будет применена MSELoss, для второй - BCELoss.

Информация о том, какая колонка должна считаться классификационной, а какая регрессионной, может быть передата через датасет: 
свойство regression_flags = [1,0]. Если в датасете не найдено данное свойство, то делается попытка рассчитать
значения regression_flags исходя из первых 1000 примеров в датасете. Если в колонке встретились только значения 0 и 1,
то она считается классификационной.

Коэффициенты шкалирования MixedCriterion получает аналогичным образом. Есди в датасете задано свойство scale = [mean, std],
то использутся данные значения, иначе производится попытка вычислить их автоматически из первых 1000 примеров датасета.
Для классификационных колонок проставляются значения mean=0 и std=1.
Если в датасете только колонки с классификацией, то эта операция просто пропускается.

В ряде случаев, например, при собственной хитрой collate_fn, автоматическое определение флагов регрессии и
коэффициентов шкалирования может привести к ошибке, даже если с данными все в порядке. В таком случае надо задать
эти параметры вручную в датасете

#### Дополнительно ####

Иногда требуется воспроизводимость обучения при повторых запусках. Для этого есть функция, определенная в pytorch_trainer.py:

```
def set_seed (seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Ее нужно вызывать до первого места, где возможен рандом. Если рандом используется в датасете (аугментации и т.д.),
то требуется вызывать ее до определения датасета.

#### Нужна особая фича? ####

Это не библиотека, а всего лишь один файл с простым шаблонным кодом. Если задача не вполне умещается в этот шаблон,
совершенно нормально - изменить код так, как требуется для задачи.


