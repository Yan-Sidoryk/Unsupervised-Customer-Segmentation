# Machine Learning II Class - Data Science Degree - NOVA IMS

## Customer Segmentation: A Key to Unlocking Business Growth and Success

In today's competitive market, understanding your customers and tailoring your marketing strategies to meet their specific needs and preferences is critical for business success. Customer segmentation, the process of dividing a large customer base into smaller groups based on shared characteristics, can help businesses gain valuable insights into their customers and develop targeted marketing strategies that maximize customer engagement and loyalty.

Unsupervised machine learning plays an important role in customer segmentation by uncovering hidden patterns in data without requiring labeled examples. Unlike traditional marketing approaches that rely on predefined customer categories, unsupervised learning allows businesses to discover natural groupings within their customer base based on purchasing behavior, demographics, browsing history, or other relevant data points.

For a data scientist, mastering unsupervised learning algorithms is essential because these techniques enable them to extract valuable insights from data without relying on predefined labels. Unlike supervised learning, which requires annotated datasets, unsupervised learning uncovers patterns, structures, and relationships within raw data, making it a powerful tool for exploratory analysis.

## The Data

For this assignment, there are two datasets containing information on customer demographics, spending habits, purchasing behavior, and historical transactions. The task is to perform customer segmentation and identify distinct groups of customers based on their shared characteristics. The two datasets are named:

- `customer_info.csv`
- `customer_basket.csv`

### `customer_info.csv` Dataset
#### Description:
Contains information about each customer, including demographics and spending behavior.

#### Columns:
- `customer_id`: Identifier of the customer.
- `customer_name`: Name of the customer (contains degree level).
- `customer_birth_date`: Birth date of the customer.
- `kids_home`: Number of kids at home.
- `teen_home`: Number of teens at home.
- `number_complaints`: Number of formal complaints by the customer.
- `location_latitude`: Approximate latitude of the customer's home.
- `location_longitude`: Approximate longitude of the customer's home.
- `distinct_stores_visited`: Number of distinct stores visited by the customer.
- `loyalty_card_number`: Number of the customer loyalty card.
- `lifetime_spend_groceries`: Total amount spent on groceries.
- `lifetime_spend_electronics`: Total amount spent on electronics.
- `lifetime_spend_vegetables`: Total amount spent on vegetables.
- `lifetime_spend_nonalcohol_drinks`: Total amount spent on non-alcoholic drinks.
- `lifetime_spend_alcohol_drinks`: Total amount spent on alcoholic drinks.
- `lifetime_spend_meat`: Total amount spent on meat.
- `lifetime_spend_fish`: Total amount spent on fish.
- `lifetime_spend_hygiene`: Total amount spent on hygiene products.
- `lifetime_spend_petfood`: Total amount spent on pet food.
- `lifetime_spend_videogames`: Total amount spent on video games.
- `lifetime_total_distinct_products`: Number of distinct products bought in lifetime.
- `year_first_transaction`: Year of first transaction.
- `percentage_of_products_bought_promotion`: Percentage of products purchased on promotion.
- `typical_hours`: Typical shopping hours.

### `customer_basket.csv` Dataset
#### Description:
Contains information about customers' different baskets purchased at the shop. Each line consists of the `customer_id`, `transaction_id`, and a list of products.

#### Columns:
- `invoice_id`: Identifier of the transaction.
- `list_of_goods`: Products bought in list format.
- `customer_id`: Identifier of the customer (used to connect with `customer_info.csv`).

## The Project

In this project, the main objectives are:

1. **Identify relevant customer segments**: Use statistical and machine learning techniques to identify meaningful segments within the customer base.
2. **Analyze customer behavior**: Gain insights into motivations, preferences, and needs by analyzing purchasing patterns, loyalty card usage, and complaint history.
3. **Develop targeted marketing strategies**: Use the `customer_basket.csv` dataset to develop personalized promotions, targeted ads, and tailored product offerings.

## Results
We performed customer segmentation using clustering techniques on retail transaction data. The clustering process yielded meaningful and well-defined segments with good evaluation scores. To deepen our understanding of each segment, we applied association rules within each cluster. This allowed us to uncover product pairings and preferences specific to each group.

Final project grade - 17/20

