# IGDB PYTHON WRAPPER

import requests
import json

class igdb:
    __api_key = ""
    __args = ""
    __api_url = "https://api-endpoint.igdb.com/"

    def __init__(self, api_key):
        self.__api_key = api_key

    # CREATE URL FROM PARAMETERS
    def joinParameters(self, parameter="", types="", default="", prefix=""):
        if parameter in self.__args:
            default = str(prefix)
            if type(self.__args[parameter]) != types:
                default += ",".join(map(str, self.__args[parameter]))
            else:
                default += str(self.__args[parameter])
        return default

    def getRequest(self,url):
        headers = {
            'user-key': self.__api_key,
            'Accept': 'application/json'
        }
        r = requests.get(self.__api_url + url, headers=headers)
        r.body = json.loads(r.text)
        return r

    # CALL TO THE API
    def call_api(self, endpoint, args):
        ids = order = filters = expand = limit = offset = search = scroll = ""
        fields = "*"
        self.__args = args

        # If dict, convert it to komma seperated string
        if type(args) != int:
            ids = self.joinParameters(parameter='ids', types=int)
            fields = self.joinParameters(parameter='fields', types=str, default="*")
            expand = self.joinParameters(parameter='expand', types=str, prefix="&expand=")
            limit = self.joinParameters(parameter='limit', types=int, prefix="&limit=")
            offset = self.joinParameters(parameter='offset', types=int, prefix="&offset=")
            order = self.joinParameters(parameter='order', types=str, prefix="&order=")
            search = self.joinParameters(parameter='search', types=str, prefix="?search=")
            scroll = self.joinParameters(parameter='scroll', types=int, prefix="&scroll=")

            if 'filters' in args:
                for key, value in args['filters'].items():
                    filters += "&filter" + key + "=" + str(value)
        else:
            ids = args

        # Build URL
        url = endpoint + "/" + str(search) + str(ids)
        url += "&" if search != "" else "?"
        url += "fields=" + str(fields) + str(filters) + str(order) + str(limit) + str(offset) + str(expand)+ str(scroll)
        print(url)
        r = self.getRequest(url)
        return r

    # Get next scroll page
    def scroll(self,response):
        r = self.getRequest(response.headers['x-next-page'])
        return r
    # GAMES
    def games(self, args=""):
        r = self.call_api("games", args)
        return r

    # PULSE
    def pulses(self, args=""):
        r = self.call_api("pulses", args)
        return r

    # CHARACTERS
    def characters(self, args=""):
        r = self.call_api("characters", args=args)
        return r

    # COLLECTIONS
    def collections(self, args=""):
        r = self.call_api("collections", args=args)
        return r

    # COMPANIES
    def companies(self, args=""):
        r = self.call_api("companies", args=args)
        return r

    # FRANCHISES
    def franchises(self, args=""):
        r = self.call_api("franchises", args=args)
        return r

    # FEEDS
    def feeds(self, args=""):
        r = self.call_api("feeds", args=args)
        return r

    # PAGES
    def pages(self, args=""):
        r = self.call_api("pages", args=args)
        return r

    # GAME_ENGINES
    def game_engines(self, args=""):
        r = self.call_api("game_engines", args=args)
        return r

    # GAME_MODES
    def game_modes(self, args=""):
        r = self.call_api("game_modes", args=args)
        return r

    # GENRES
    def genres(self, args=""):
        r = self.call_api("genres", args=args)
        return r

    # KEYWORDS
    def keywords(self, args=""):
        r = self.call_api("keywords", args=args)
        return r

    # PEOPLE
    def people(self, args=""):
        r = self.call_api("people", args=args)
        return r

    # PLATFORMS
    def platforms(self, args=""):
        r = self.call_api("platforms", args=args)
        return r

    # PLAYER_PERSPECTIVES
    def player_perspectives(self, args=""):
        r = self.call_api("player_perspectives", args=args)
        return r

    # RELEASE_DATES
    def release_dates(self, args=""):
        r = self.call_api("release_dates", args=args)
        return r

    # PULSE GROUPS
    def pulse_groups(self, args=""):
        r = self.call_api("pulse_groups", args=args)
        return r

    # PULSE SOURCES
    def pulse_sources(self, args=""):
        r = self.call_api("pulse_sources", args=args)
        return r

    # THEMES
    def themes(self, args=""):
        r = self.call_api("themes", args=args)
        return r

    # REVIEWS
    def reviews(self, args=""):
        r = self.call_api("reviews", args=args)
        return r

    # TITLES
    def titles(self, args=""):
        r = self.call_api("titles", args=args)
        return r

    # CREDITS
    def credits(self, args=""):
        r = self.call_api("credits", args=args)
        return r
