import requests

url = "https://www.medicinalplants.in/siddhasearchpage"

html = requests.get(url).text

index = html.find("function processing")
print(html[index:index+2500])