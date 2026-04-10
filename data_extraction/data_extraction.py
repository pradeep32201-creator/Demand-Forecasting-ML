import pandas as pd

from sqlalchemy import create_engine


def load_data():
    conn_str = "postgresql://ecom_ro_user.imnzftquwjuxcwpeufwp:work-experience-read-only@aws-0-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require"
    engine = create_engine(conn_str)

    query = '''
        SELECT
            DATE(created_at) as date,
            variant_id,
            SUM(qty) as quantity,
            SUM(qty * unit_price) as total_revenue
        FROM ecom.orders o
        JOIN ecom.order_items oi ON o.order_id = oi.order_id
        GROUP BY DATE(created_at),variant_id
    '''
    df = pd.read_sql(query, engine)
    return df







