import requests
import pandas as pd
import time

def get_product_name(product_id):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    url = f"https://tiki.vn/api/v2/products/{product_id}"
    resp = requests.get(url, headers=headers)

    if resp.status_code == 200:
        data = resp.json()
        return data.get("name", "Unknown")
    else:
        print(f"Lỗi khi lấy tên sản phẩm: {resp.status_code}")
        return "Unknown"

def get_reviews(product_id, max_reviews=200):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    reviews = []
    page = 1

    while len(reviews) < max_reviews:
        url = f"https://tiki.vn/api/v2/reviews?product_id={product_id}&limit=20&page={page}&sort=score|desc"
        print(f"Crawling page {page}: {url}")
        resp = requests.get(url, headers=headers)

        if resp.status_code != 200:
            print(f"Lỗi {resp.status_code}")
            break

        data = resp.json()
        if "data" not in data or not data["data"]:
            print("Không còn review nào.")
            break

        for item in data["data"]:
            content = item.get("content", "")
            if content:
                title = item.get("title", "")
                rating = item.get("rating", 0)
                review_id = item.get("id")
                reviews.append([review_id, content, title, rating])
                if len(reviews) >= max_reviews:
                    break

        page += 1
        time.sleep(1)

    df = pd.DataFrame(reviews, columns=["Review_ID", "Content", "Title", "Rating"])
    return df
