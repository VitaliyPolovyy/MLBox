import pyodbc

def execute_sql_mssql(query: str) -> str:
    try:
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=10.11.104.24,1433;"
            "DATABASE=TEST;"
            "UID=polovyy;"
            "PWD=zlQgut5e",
            timeout=5
        )
        cursor = conn.cursor()
        cursor.execute(query)

        if cursor.description:
            rows = cursor.fetchall()
            result = [tuple(row) for row in rows]
        else:
            conn.commit()
            result = f"{cursor.rowcount} row(s) affected."

        conn.close()
        return str(result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print(execute_sql_mssql("SELECT TOP 1 name FROM sys.databases"))
