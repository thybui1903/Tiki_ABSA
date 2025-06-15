import requests
import time
from src.data_building.config import HEADERS, SLEEP_TIME # Import từ config

def get_products_in_category(category_id, num_pages):
    """Lấy danh sách sản phẩm trong một danh mục cụ thể."""
    products = []
    for page in range(1, num_pages + 1):
        url = f"https://tiki.vn/api/personalish/v1/blocks/listings?limit=40&category={category_id}&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('data', []):
                products.append({
                    'name': item.get('name'),
                    'price': item.get('price'),
                    'id': item.get('id'),
                    'spid': item.get('sku'),
                    'url': item.get('url_path')
                })
        else:
            print(f"⚠️ Lỗi truy cập danh mục {category_id}, trang {page}")
        time.sleep(SLEEP_TIME)
    return products

def get_reviews_for_product(spid, product_id, num_pages_per_product):
    """Lấy bình luận cho một sản phẩm cụ thể."""
    reviews_data = []
    for page in range(1, num_pages_per_product + 1):
        url = (
            f"https://tiki.vn/api/v2/reviews?limit=5&include=comments,"
            f"contribute_info,attribute_vote_summary&sort=score%7Cdesc,id%7Cdesc,"
            f"stars%7Call&page={page}&spid={spid}&product_id={product_id}&seller_id=1"
        )
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"⚠️ Lỗi trang {page} của SPID {spid}")
            break

        reviews = response.json().get("data", [])
        if not reviews:
            break

        for review in reviews:
            comment_id = review.get('id')
            content = review.get('content', '').strip()
            title = review.get('title', '')
            rating = review.get('rating')

            if content:
                reviews_data.append({
                    "id": comment_id,
                    "title": title if isinstance(title, str) else '',
                    "content": content,
                    "rating": rating
                })
        time.sleep(SLEEP_TIME)
    return reviews_data