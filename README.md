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
export OPENAI_API_KEY=[your open ai key goes here without the square brackets]

then run this
``` bash
source source.env
```

``` bash
chainlit run app.py
```
