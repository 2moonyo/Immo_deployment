## CODE

- why 2 readme? 
- commits messages don't make sense ("deploy changes", "directory",...)
- main_model_train.py almost 600 lines of code, no OOP logic at all it's difficult to read
- same remark for deploy_main.py
- in utils usually we put python files and not data, you could create a "data" folder with specific subfolders (geospatial,...)
- and then for OOP logic, find a way to split your 2 monolithics scripts into modules with unit logic and modify the architecture accordingly (utils/train, utils/deploy)
- Code can be optimized, mappings could be json files instead of variables, functions could be added to avoid code duplicate,...
- streamlit entry script should be "app.py"
- training entry point could be "main.py" or "training_main.py" in that specific case
- make sure you add typing and docstring 
- you can pimp a bit the readme, add some visuals,... 
- I do understand it's an old project, that's why I give you all the possible improvements but at that time this was a good one :fire:
- But if you want to polish your portfolio, you may consider implementing my remarks. the idea is to make it easy to understand and modify to someone else :) 