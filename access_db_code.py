import sqlite3
import pandas as pd

database_name = "teacher_llm_dataset_2000.db"

# Step 2: Connect to your database
# Replace 'your_database.db' with the path to your actual database file
conn = sqlite3.connect(database_name)

# Step 3: Query the database
# Replace 'your_table_name' with the name of the table you want to view
query = "SELECT * FROM LLM_results"

# Step 4: Load the query results into a Pandas DataFrame
df = pd.read_sql_query(query, conn)

# Don't forget to close the connection when you're done
conn.close()
