from src.crawl.crawler import crawl_all_products
from src.crawl.product_ids import product_ids

if __name__ == "__main__":
    crawl_all_products(product_ids)