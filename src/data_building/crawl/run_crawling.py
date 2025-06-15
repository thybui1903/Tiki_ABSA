import json
from src.data_building.config import CATEGORY_IDS, MAX_PER_CATEGORY, OUTPUT_FILE_NAME, CRAWL_PRODUCT_PAGES, CRAWL_REVIEW_PAGES_PER_PRODUCT
from src.data_building.crawl.tiki_crawler import get_products_in_category, get_reviews_for_product

def main_crawl():
    all_comments = 0
    max_total = MAX_PER_CATEGORY * len(CATEGORY_IDS)

    with open(OUTPUT_FILE_NAME, mode="w", encoding="utf-8") as file:
        for cat_id in CATEGORY_IDS:
            print(f"▶️ Bắt đầu crawl danh mục {cat_id}")
            category_comments = 0
            products = get_products_in_category(cat_id, CRAWL_PRODUCT_PAGES)

            for product in products:
                if category_comments >= MAX_PER_CATEGORY or all_comments >= max_total:
                    break

                spid = product.get('spid')
                product_id = product.get('id')

                if not spid or not product_id:
                    continue

                reviews = get_reviews_for_product(spid, product_id, CRAWL_REVIEW_PAGES_PER_PRODUCT)

                for review in reviews:
                    if category_comments >= MAX_PER_CATEGORY or all_comments >= max_total:
                        break
                    json.dump(review, file, ensure_ascii=False)
                    file.write('\n')

                    category_comments += 1
                    all_comments += 1

            print(f"✅ Hoàn tất danh mục {cat_id}: {category_comments} bình luận\n")

    print(f"🎉 Đã crawl tổng cộng {all_comments} bình luận từ {len(CATEGORY_IDS)} danh mục!")

if __name__ == "__main__":
    main_crawl()