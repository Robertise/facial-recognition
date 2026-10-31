import sqlite3

conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Xóa theo id
cursor.execute("DELETE FROM employees WHERE id = ?", ("NV002",))

# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()