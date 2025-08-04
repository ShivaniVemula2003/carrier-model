import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Database connection
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="zippyy_orders",
    user="postgres",
    password="shivani@123"
)

# Query shipments table
query = """
SELECT 
  carrier_name,
  customer_pincode AS destination_pincode,
  '500075' AS origin_pincode,  -- Simulated origin pincode (use real if available)
  shipment_status
FROM shipments
WHERE shipment_status IN ('pre_transit', 'in_transit');
"""

df = pd.read_sql_query(query, conn)
print(f"✅ Rows returned from DB: {len(df)}")
print(df.head())

# Close DB connection
conn.close()

# Drop rows with missing important data
df.dropna(subset=['carrier_name', 'destination_pincode', 'origin_pincode', 'shipment_status'], inplace=True)

# Simulate RTO vs Delivered status (temporary logic for demo)
# Let's say 'pre_transit' = success, 'in_transit' = RTO
df['delivered'] = df['shipment_status'].apply(lambda x: 1 if x == 'pre_transit' else 0)

# Encode pincodes and carrier
le_origin = LabelEncoder()
le_dest = LabelEncoder()
le_carrier = LabelEncoder()

df['origin_encoded'] = le_origin.fit_transform(df['origin_pincode'].astype(str))
df['dest_encoded'] = le_dest.fit_transform(df['destination_pincode'].astype(str))
df['carrier_encoded'] = le_carrier.fit_transform(df['carrier_name'].astype(str))

# Prepare features and target
X = df[['origin_encoded', 'dest_encoded', 'carrier_encoded']]
y = df['delivered']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Predict delivery % for carrier per pincode pair
df_grouped = df.groupby(['origin_pincode', 'destination_pincode', 'carrier_name'])['delivered'].agg(['sum', 'count']).reset_index()
df_grouped['delivery_percent'] = (df_grouped['sum'] / df_grouped['count']) * 100

print("\n📍 Example Carrier Delivery % Rates:")
print(df_grouped[['origin_pincode', 'destination_pincode', 'carrier_name', 'delivery_percent']].head(10))


