import pandas as pd
from src.config import MAX_REVIEWS_PER_PRODUCT, RAW_DATA_PATH
from src.crawl.tiki_api import get_reviews
from src.crawl.product_ids import product_ids

def crawl_all_products(product_ids, max_reviews=MAX_REVIEWS_PER_PRODUCT, output_path=RAW_DATA_PATH):
    all_reviews_df = pd.DataFrame(columns=["Review_ID", "Content", "Title", "Rating"])

    for product_id in product_ids:
        df = get_reviews(product_id, max_reviews)
        all_reviews_df = pd.concat([all_reviews_df, df], ignore_index=True)
        print(f"✅ Đã crawl xong product_id: {product_id}")

    all_reviews_df.to_json(output_path, orient="records", force_ascii=False, indent=4)
    print(f"\n✅ Dữ liệu đã lưu tại: {output_path}")