Deploy instructions to run chainlit
``` bash
git clone https://github.com/mileslopesny/demoday.git
```

``` bash
cd demoday
```

``` bash
pip install -r ewquirements.txt
```

create a file called source.env with this line : 
exprt OPENAI_API_KEY=[your open ai key goes here]

then run this
``` bash
source source.env
```

``` bash
chainlit run app.py
```
