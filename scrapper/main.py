import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

def scrape_shl_catalog():
    """
    Scrape product catalog data from SHL website
    """
    base_url = "https://www.shl.com"
    start_url = "https://www.shl.com/products/product-catalog/"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        def get_page(url_to_fetch: str) -> BeautifulSoup:
            """Fetch a page and return a BeautifulSoup object."""
            print(f"Fetching page: {url_to_fetch}")
            resp = requests.get(url_to_fetch, headers=headers, timeout=10)
            resp.raise_for_status()
            return BeautifulSoup(resp.content, 'html.parser')

        # track visited pages to avoid infinite loops
        seen_urls = set()
        page_url = start_url
        data = []

        def scrape_product_detail(relative_path: str) -> dict:
            """Fetch additional metadata from the individual product page."""
            detail = {
                'Description': '',
                'Job Levels': '',
                'Languages': '',
                'Assessment Length': '',
                'Downloads': ''
            }
            if not relative_path:
                return detail
            # resolve against base_url rather than undefined `url`
            detail_url = urljoin(base_url, relative_path)
            try:
                resp = requests.get(detail_url, headers=headers, timeout=10)
                resp.raise_for_status()
            except requests.RequestException:
                print(f"  ⚠️ Failed to fetch detail page: {detail_url}")
                return detail
            dsoup = BeautifulSoup(resp.content, 'html.parser')
            # look for rows in the product details
            rows = dsoup.find_all('div', class_='product-catalogue-training-calendar__row')
            for r in rows:
                # heading h4
                heading = r.find('h4')
                if not heading:
                    continue
                key = heading.get_text(strip=True)
                # get the paragraph text
                p = r.find('p')
                value = p.get_text(separator=' ', strip=True) if p else ''
                if key == 'Description':
                    detail['Description'] = value
                elif key == 'Job levels':
                    detail['Job Levels'] = value
                elif key == 'Languages':
                    detail['Languages'] = value
                elif key == 'Assessment length':
                    detail['Assessment Length'] = value
                elif key == 'Downloads':
                    # collect link texts and hrefs
                    downloads = []
                    for li in r.select('ul.product-catalogue__downloads li'):
                        a = li.find('a')
                        if a:
                            text = a.get_text(strip=True)
                            href = a.get('href')
                            downloads.append(f"{text} ({href})")
                    detail['Downloads'] = '; '.join(downloads)
            # debug print the metadata retrieved for this product
            if any(detail.values()):
                print(f"    -> metadata fetched: {detail}")
            return detail
        
        # iterate through each paginated page until there is no next link
        while page_url and page_url not in seen_urls:
            seen_urls.add(page_url)
            soup = get_page(page_url)

            # Find ALL table containers (page 1 has 2, page 2+ has 1)
            table_containers = soup.find_all('div', class_='custom__table-responsive')
            if not table_containers:
                print(f"Error: Could not find any table containers on {page_url}")
                break

            print(f"Found {len(table_containers)} table(s) on {page_url}")

            # Process each table container
            for container_idx, table_container in enumerate(table_containers):
                # Find the table
                table = table_container.find('table')
                if not table:
                    print(f"  Warning: Could not find table inside container {container_idx + 1}")
                    continue

                # Extract table category from the header
                header_th = table.find('th', class_='custom__table-heading__title')
                table_category = header_th.get_text(strip=True) if header_th else 'Unknown'

                # Only process "Individual Test Solutions" tables, skip others
                if table_category != 'Individual Test Solutions':
                    print(f"  Skipping table: {table_category}")
                    continue

                # Extract data rows
                rows = table.find_all('tr')
                print(f"  Table ({table_category}): Found {len(rows) - 1} data rows")

                for row in rows[1:]:
                    tds = row.find_all('td')

                    if len(tds) < 4:
                        continue

                    # Extract product name from the link
                    product_link = tds[0].find('a')
                    product_href = ''
                    if product_link:
                        product_name = product_link.get_text(strip=True)
                        product_href = product_link.get('href')
                    else:
                        product_name = tds[0].get_text(strip=True)

                    # Extract Remote Testing (check if "-yes" class exists)
                    remote_testing = "Yes" if tds[1].find('span', class_='catalogue__circle -yes') else "No"

                    # Extract Adaptive/IRT (check if "-yes" class exists)
                    adaptive_irt = "Yes" if tds[2].find('span', class_='catalogue__circle -yes') else "No"

                    # Extract Test Type (collect all product-catalogue__key spans)
                    test_keys = []
                    test_type_cell = tds[3]
                    test_spans = test_type_cell.find_all('span', class_='product-catalogue__key')
                    for span in test_spans:
                        test_keys.append(span.get_text(strip=True))

                    test_type = ', '.join(test_keys) if test_keys else ""

                    # fetch detail page metadata
                    metadata = scrape_product_detail(product_href)
                    
                    # resolve product URL against base_url
                    product_url = urljoin(base_url, product_href) if product_href else ''

                    row_data = {
                        'Category': table_category,
                        'Product Name': product_name,
                        'Product URL': product_url,
                        'Remote Testing': remote_testing,
                        'Adaptive/IRT': adaptive_irt,
                        'Test Type': test_type,
                        'Description': metadata.get('Description', ''),
                        'Job Levels': metadata.get('Job Levels', ''),
                        'Languages': metadata.get('Languages', ''),
                        'Assessment Length': metadata.get('Assessment Length', ''),
                        'Downloads': metadata.get('Downloads', '')
                    }
                    data.append(row_data)

                    print(f"  ✓ Extracted: {product_name}")

            # pagination - look for the SECOND pagination (skip the first one on page 1)
            # page 1 has 2 paginations, page 2+ has only 1
            paginations = soup.find_all('ul', class_='pagination')
            next_url = None
            if paginations:
                # Use the last pagination element (handles both cases: 2 on page 1, 1 on page 2+)
                pagination = paginations[-1]
                next_li = pagination.find('li', class_='pagination__item -arrow -next')
                if next_li:
                    a = next_li.find('a')
                    if a and a.get('href'):
                        next_url = urljoin(base_url, a['href'])
            if next_url and next_url not in seen_urls:
                page_url = next_url
                print(f"➡️ moving to next page: {page_url}")
                continue
            else:
                break
        
        # Write to JSON
        if data:
            json_filename = 'shl_product_catalog.json'
            print(f"\nWriting {len(data)} records to {json_filename}...")
            
            with open(json_filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=2, ensure_ascii=False)
            
            print(f"✓ Successfully saved data to {json_filename}")
            print(f"Total products scraped: {len(data)}")  
        else:
            print("No data was extracted from the table")
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    scrape_shl_catalog()
