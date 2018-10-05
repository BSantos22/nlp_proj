## This repo is no longer officially supported though it's still open to contributions via pull requests and issue support.

[![Build Status](https://travis-ci.org/igdb/igdb_api_python.svg?branch=master)](https://travis-ci.org/igdb/igdb_api_python)
[![Coverage Status](https://coveralls.io/repos/github/igdb/igdb_api_python/badge.svg?branch=master)](https://coveralls.io/github/igdb/igdb_api_python?branch=master)
[![PyPI version](https://badge.fury.io/py/igdb-api-python.svg)](https://badge.fury.io/py/igdb-api-python)

It is expected that you know how the API works and read the documentation: https://igdb.github.io/api/about/welcome/

-------------

To use this wrapper, run:

`pip install igdb_api_python`

Example on how to use it

```python
from igdb_api_python.igdb import igdb

igdb = igdb("YOUR_KEY")
result = igdb.games(1942)

for game in result.body:
    print("Retrieved: " + game["name"])
```
`result` is an requests object: http://docs.python-requests.org/en/master/  
So you can check on status_code etc.

[Example.py](https://github.com/igdb/igdb_api_python/blob/master/example.py) for more examples

"# nlp_proj" 
