import requests

def check_websites(websites):
    accessible = []
    inaccessible = []
    
    for url in websites:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"[Accessible] {url}")
                accessible.append(url)
            else:
                print(f"[Inaccessible] {url} - Status Code: {response.status_code}")
                inaccessible.append(url)
        except requests.ConnectionError:
            print(f"[Inaccessible] {url} - Connection Error")
            inaccessible.append(url)
        except requests.Timeout:
            print(f"[Inaccessible] {url} - Timeout")
            inaccessible.append(url)
        except requests.RequestException as e:
            print(f"[Inaccessible] {url} - Error: {e}")
            inaccessible.append(url)
    
    return accessible, inaccessible

# List of websites to check
websites_to_check = [
"https://python.org",
"https://pypi.org",
"https://huggingface.co",
"https://google.com",
"https://colab.research.google.com",
"https://docker.com",
"https://hub.docker.com",
"https://podman.io",
"https://github.com",
"https://kaggle.com"
]

accessible_sites, inaccessible_sites = check_websites(websites_to_check)

print("\nSummary:")
print("Accessible Websites:", accessible_sites)
print("Inaccessible Websites:", inaccessible_sites)
