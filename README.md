# Week 6: Interactive Sales Dashboard

This repository contains a professional business intelligence and data analytics dashboard project that analyzes sales performance, product trends, customer behavior, and regional distributions in Python. 

The dashboard leverages **Seaborn** and **Matplotlib** for high-fidelity statistical plots, and **Plotly** for responsive, browser-ready interactive charts.

---

## 📂 Project Structure

```
week6/
├── sales_data.csv             # Source transactions dataset (existing)
├── requirements.txt           # Python library dependencies
├── dashboard.ipynb            # Interactive Jupyter Notebook analysis
├── dashboard.py               # Standalone pipeline script (generates assets)
├── README.md                  # Comprehensive project documentation
├── analysis_report.pdf        # Professional executive PDF report
├── dashboard_demo.gif         # Animated demo of the interactive Plotly charts
└── visualizations/            # Saved static & interactive visualization folder
    ├── box_plot.png
    ├── violin_plot.png
    ├── correlation_heatmap.png
    ├── region_count_plot.png
    ├── sales_histogram.png
    ├── multi_plot_dashboard.png
    ├── plotly_bar.html        # Interactive Bar Chart
    ├── plotly_line.html       # Interactive Line Chart
    ├── plotly_pie.html        # Interactive Pie Chart
    └── plotly_scatter.html    # Interactive Price vs Quantity Scatter
```

---

## 🎯 Project Objectives

1. **Perform Exploratory Data Analysis (EDA):** Inspect structure, data types, completeness, and descriptive statistics of the sales transactions.
2. **Execute Data Cleaning & Feature Engineering:** Prepare the dataset by converting date strings and deriving calendar attributes (Year, Month, Month Name, Day of Week, Quarter).
3. **Analyze Statistical Distributions:** Uncover patterns in unit prices and transaction values across different product lines and geographical regions using Seaborn.
4. **Develop Interactive Web Visualizations:** Build responsive, hover-enabled, zoomable visualization interfaces using Plotly.
5. **Formulate Actionable Business Insights:** Answer core strategic business performance questions to guide resource and inventory allocation.
6. **Compile Executive Business Report:** Export findings and plots into a formatted PDF document.

---

## 📊 Dataset Description

The analysis is based on `sales_data.csv` containing transactions with the following fields:

* **Date:** Timestamp of transaction (formatted as `YYYY-MM-DD`).
* **Product:** Product category sold (`Laptop`, `Phone`, `Tablet`, `Monitor`, `Headphones`).
* **Quantity:** Number of units purchased.
* **Price:** Unit price in USD.
* **Customer_ID:** Unique customer identifier.
* **Region:** Geographical sale territory (`East`, `North`, `West`, `South`).
* **Total_Sales:** Gross transaction value (`Quantity * Price`).

---

## 🛠️ Setup & Installation Instructions

To set up the project on your local machine and run the scripts:

### Prerequisites
* Python 3.8 or higher installed.

### Steps

1. **Clone or navigate to the directory:**
   ```bash
   cd week6
   ```

2. **Install dependencies:**
   Install all required libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Dashboard Assets and PDF Report:**
   To run the pipeline and build all visual deliverables:
   ```bash
   # Run data processing, save visualizations & print stats
   python dashboard.py
   ```
   *Note: If you wish to re-generate the PDF report layout or change code styles, the generated code files are located in your workspace.*

---

## 📈 Visualizations Explanation & Business Interpretation

### 1. Statistical Seaborn Visualizations
* **Box Plot (Price Distribution by Product):** Highlights that `Laptop` and `Phone` are premium-tier categories with medians above $30k, while `Monitor` and `Headphones` represent budget/entry-level items.
* **Violin Plot (Sales Value by Region):** Exhibits the density curve of order values. The South and West regions show a wider distribution shape at higher revenue ranges.
* **Correlation Heatmap:** Illustrates that order values are heavily correlated with unit prices. No meaningful correlation exists between date attributes and sales, indicating stable customer demand.
* **Count Plot (Transaction Frequency by Region):** Reveals that transaction counts are relatively uniform, with the `North` and `South` leading by a small margin.
* **Histogram with KDE (Transaction Values):** Displays a highly right-skewed profile, with the majority of orders falling under $150k, and occasional large B2B orders stretching to $350k+.

### 2. Interactive Plotly Visualizations (saved in `visualizations/`)
* **`plotly_bar.html` (Sales Revenue by Product):** An interactive bar chart detailing total revenue share. Laptops contribute the highest absolute sales.
* **`plotly_line.html` (Daily Sales Revenue Trend):** Shows revenue fluctuations across early 2024. Users can hover, pan, and zoom to view specific dates.
* **`plotly_pie.html` (Revenue Share by Region):** Donut chart showing the South and North control the largest portion of sales revenue.
* **`plotly_scatter.html` (Price vs. Quantity Sold):** Scatter plot showing how transactions aggregate by quantity and price.

---

## 💡 Key Business Answers & Strategic Recommendations

* **Which products generate the highest sales?**
  Laptops lead overall sales revenue ($3.89M), followed by Tablets ($2.88M) and Phones ($2.86M). Laptops represent the premium revenue anchor.
* **Which regions perform best?**
  The **North** region performs best, contributing $3.98M in total sales, followed by the **South** ($3.74M).
* **Are there noticeable outliers?**
  Yes. Single bulk transactions of high volume exist in the dataset (such as Headphones sales of $349,510 in the West region). These represent high-margin corporate B2B contracts.
* **Strategic Recommendations:**
  1. **Optimize Shipping & Warehousing:** Focus logistics resources on the North and South regions to decrease shipping times on high-demand Laptop/Phone models.
  2. **Product Bundling:** Bundle high-margin, low-price accessories (Monitors, Headphones) with Laptops and Phones to increase average order values.
  3. **Enterprise Marketing:** Deploy dedicated B2B sales campaigns in the West and East regions to secure more high-value bulk corporate purchases.

---

## 🧪 Testing and Validation

All scripts contain print statements and assert blocks to validate:
* The input file `sales_data.csv` is correctly read.
* Clean data is formatted properly without null inputs.
* The visualizations directory exists and output files are saved without issues.
* The PDF report canvas computes correct total page count dynamically.
