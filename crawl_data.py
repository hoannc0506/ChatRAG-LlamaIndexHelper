from bs4 import BeautifulSoup
import requests
import os
from tqdm import tqdm
import fire

# Extract and print the main text from each card
def get_blog_urls():
    # get all blog
    base_url = "https://www.llamaindex.ai"
    response = requests.get(f"{base_url}/blog")
    html = response.text
    
    soup = BeautifulSoup(html, 'html.parser')
    # Find all blog post cards
    blog_cards = soup.find_all('div', class_='CardBlog_card__mm0Zw')
    blog_data = []
    for card in blog_cards:
        # Extract title
        title_element = card.find('p', class_='CardBlog_title__qC51U').find('a')
        title = title_element.text.strip()
        url = base_url + title_element['href']
    
        # Extract publication date
        date = card.find('p', class_='Text_text__zPO0D Text_text-size-16__PkjFu').text.strip()
    
        # Print the extracted information
        print(f"Title: {title}")
        print(f"Date: {date}")
        print(f"URL: {url}")
        print("---")

        blog_data.append({
            "title": title,
            "date": date,
            "url": url
        })
        
    return blog_data

def extract_page(page_url):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script and style tags
    for script in soup(['script', 'style']):
        script.extract()

    # Get rid of empty tags
    for tag in soup.find_all():
        if not tag.text.strip():
            tag.extract()

    # only get main blog content, ignore related blogs
    blog_content = soup.find("main").find("div", class_="BlogPost_htmlPost__Z5oDL")

    return blog_content.prettify()


def main(save_folder):
    # extract all blog page and save to folder data
    blogs_data = get_blog_urls()

    for blog_data in tqdm(blogs_data,desc="Crawling data"):
        cleaned_html = extract_page(blog_data["url"])
        save_name =  blog_data["url"].split("/")[-1]
        with open(f'{save_folder}/{save_name}.html', 'w', encoding='utf-8') as f:
            f.write(cleaned_html)

if __name__ == "__main__":
    fire.Fire(main)