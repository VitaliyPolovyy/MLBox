import pyodbc 

def test_sql_server_connection():
    # Parse your connection string
    connection_string = (
        "Driver={SQL Server};"
        "Provider=SQLOLEDB;"
        "DATABASE=prod;"
        "UID=kiev_1c_exchange;"
        "PWD=mg50Yxrv;"
        "Server=plvs-itebd.r.roshen.com"
    )
    
    try:
        # Attempt to connect
        conn = pyodbc.connect(connection_string)
        print("✅ Connection successful!")
        
        # Test with a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        print(f"Query test result: {result[0]}")
        
        # Close connection
        conn.close()
        return True
        
    except pyodbc.Error as e:
        print(f"❌ Connection failed: {e}")
        return False

# Run the test
test_sql_server_connection()