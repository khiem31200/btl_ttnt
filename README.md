# Bài tập lớn trí tuệ nhân tạo nhóm 2
## Nhóm sinh viên thực hiện

### Nguyễn Hòa Khiêm - 20252031M
### Phạm Hữu Ngân	- 20252041M
### Bùi Phạm Sơn Hà	- 20251053M


## Cài đặt

### 1. IDE: Cài đặt Visual Studio Code và các extension hỗ trợ cho python
### 2. Cài đặt python version 3.13 trở xuống
### 3. Cài đặt Anaconda: Khởi chạy trong terminal (Windows)
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o
.\miniconda.exe
start /wait "" .\miniconda.exe /S
del .\miniconda.exe
```
### 4. Pull source code về
### 5. Truy cập vào thư mục dự án (Nếu chưa có thì tạo thư mục dự án và đẩy source code vừa pull vào đó) và cài các thư viện 
- Trong Visual Studio Code nhấn tổ hợp `CTRL + ~ ` sau đó chạy câu lệnh
```
conda create --prefix ./venv python=3.12.3
```
- Trong Visual Studio Code nhấn tổ hợp `CTRL+SHIFT+P` để chọn phiên bản của Python `3.12.3 conda`, khi thành công sẽ thấy version của môi trường conda được chọn
  
    <img width="588" height="39" alt="image" src="https://github.com/user-attachments/assets/d55e75cd-9fef-4830-b135-15f0478cd255" />

- Mở terminal (Cần trỏ vào đúng thư mục dự án)
  + Có thể sử dụng terminal trực tiếp trong Visual Studio Code
  + Hoặc sử dụng Anaconda Propt
    
    <img width="733" height="533" alt="image" src="https://github.com/user-attachments/assets/eed87aac-a191-45ed-a5fd-d8bf6a9f8860" />

- Cài đặt các thư viện `osmnx` `folium` `streamlit`, nếu có yêu cầu cập nhật hãy chon yes (y)
```
conda install -c conda-forge osmnx folium streamlit
```

### 6. Cài đặt bản đồ (Nếu không có)
- Chạy lệnh sau 
```
python generate_map.py
```

- Gán tên file vừa tạo vào file dự án của bạn, ví dụ như dưới hãy gán tên file `hanoi_new.graphml` vào file dự án.
```
output_filename = 'hanoi_new.graphml'
```

## Khởi chạy
- Yêu cầu phải có internet
- Mở terminal (Áp dụng mục `Mở terminal` phía trên) và chạy câu lệnh sau
```
streamlit run <Tên file>.py
```
- Truy cập cập port ở trên terminal để truy cập giao diện, thường sẽ là 8501 -> Host sẽ là http://localhost:8501
