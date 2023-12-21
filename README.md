# Прикладное программирование

## Состав лабораторных работ:
[Лабораторная 1](https://github.com/itsecd/python/tree/main/Lab1) - Автоматизированный сбор данных. Работа со строками;  
[Лабораторная 2](https://github.com/itsecd/python/tree/main/Lab2) - Обработка данных;  
[Лабораторная 3](https://github.com/itsecd/python/tree/main/Lab3) - Работа с GUI;  
[Лабораторная 4](https://github.com/itsecd/python/tree/main/Lab4) - Анализ и визуализация данных;  
[Лабораторная 5](https://github.com/itsecd/python/tree/main/Lab5) - Обучение и тестирование модели.

Текущая успеваемость доступна [здесь](https://docs.google.com/spreadsheets/d/1-T5bN60sduQSYTlml_Gg0qIEoQLg-EZVz1_b6QNdtjE/edit?usp=sharing).

## Правила работы с кодом:
1. Для выполнения лабораторной работы сначала *необходимо* [форкнуть](https://docs.github.com/en/get-started/quickstart/fork-a-repo) этот репозиторий;  
1.1. Имя форкнутого репозитория может быть каким вам удобно;  
1.2. В дескрипшене репозитория будет полезно указать свои ФИО, номер группы;  
1.3. Копировать необходимо `main` ветку;  
1.4. Для выполнения лабораторной работы  необходимо в форкнутом репозитории [завести новую ветку](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository), основанную на вашей `main`-ветке;   
1.5. Ветка *должна* именоваться `lab*номер_лабы*-var*номер_варианта*`. Все модификации репозитория для указанной лабораторной работы должны производиться именно в этой ветке;


2. [Склонировать](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) полученный репозиторий себе на машину.


3. Создать проект в выбранной IDE, начать реализовывать необходимый по заданию код и [коммитить](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/about-commits) результаты;  
3.1. Преподаватель с большей вероятностью поверит в то, что код в репозитории написан вами лично, если коммитов в репозитории будет больше одного;  
3.2. Так как лабораторные в целом небольшие, было бы неплохо делать коммит сразу, как только решена некоторая промежуточная задача;  
3.3. Коммиты *должны* иметь вменяемые описания на английском языке;  
3.4. Ваш репозиторий *должен* содержать файлы [.gitignore](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files) (для них имеется набор [шаблонов](https://github.com/github/gitignore)) и [requirements.txt](https://www.jetbrains.com/help/pycharm/managing-dependencies.html#create-requirements);  


4. Когда необходимый код написан, соответствует [PEP](https://peps.python.org/pep-0008/) и **работает**, необходимо сделать [пул-риквест в исходный репозиторий](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork);  
4.1. Соответственно, в параметрах пул-риквеста в качестве `base repository` указывается `itsecd\python`, в качестве `base` - `main`;  
4.2. В качестве `head repository` указывается ваш форкнутый репозиторий, в качестве `compare` - ветка, в которой вы работали в своем репозитории;  
4.3. Название пул риквеста *должно* формироваться следующим образом: `<Номер группы> <ФИО> Лаб.<Номер лабы> Вар.<Номер варианта>`;  
4.4. В описание пул риквеста было бы неплохо добавить описание того, что  от вас требовалось сделать в данной лабораторной согласно вашему варианту. Хотя бы на русском языке;  
4.5. Преподаватель поставит себя в [ревьюеры](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review), назначит вас [исполнителем](https://docs.github.com/en/issues/tracking-your-work-with-issues/assigning-issues-and-pull-requests-to-other-github-users) и произведет ревью;  


5. После успешного создания пул-риквеста производится ревью кода лабораторной;  
5.1. Если к работоспособности и внешнему виду кода нет претензий, то преподаватель аппрувит и закрывает пул риквест;  
5.2. Если претензии к коду есть, они указываются в ревью. После их устранения в рамках текущего пул риквеста, вам необходимо запросить [повторное ревью](https://github.blog/changelog/2019-02-21-re-request-review-on-a-pull-request/);  


6. Поздравляю, практическая часть лабораторной принята, можно переходить к теоретической.


7. Для выполнения следующей лабораторной работой заводите в своем репозитории очередную `lab*номер_лабы*-var*номер_варианта*` ветку, основанную на `main`.


## Правила сдачи лабораторных:
Для успешной сдачи лабораторной работы необходимо:
1. Успешно закрыть пул-риквест;
2. Ответить на вопросы по коду;
3. Ответить на теоретические вопросы.  

Количество и качество задаваемых вопросов может варьироваться.

## Ремарки:
Работать с git вы можете так, как вам удобно:
* через интерфейс [командной строки](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git),
* через [плагин](https://www.jetbrains.com/help/pycharm/set-up-a-git-repository.html#fetch) в IDE,
* через десктопный клиент, типа [такого](https://desktop.github.com/).

Если вы столкнулись с непреодолимыми трудностями в ходе выполнения лабораторной работы, вы можете задать вопрос в:
* дискорд-канале, посвященном предмету,
* телеграм чате вашего курса.


