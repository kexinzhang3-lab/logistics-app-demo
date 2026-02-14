# 🚛 Logistics Decision Support System

A comprehensive operations research platform integrating vehicle routing, inventory optimization, and facility location planning.

---

## 🎯 Core Capabilities

### 1. Vehicle Routing Problem (VRP) 车辆路径规划
- Optimize delivery routes for minimal total distance
- Support for both coordinate-based and distance-matrix inputs
- Open VRP mode (no return to depot)

### 2. Quantity Discount EOQ Model 数量折扣经济订货批量
- Dynamic pricing tier analysis
- Minimizes total annual cost (purchase + setup + holding)
- Bilingual support (Chinese/English)

### 3. Facility Location Optimization 工厂选址优化
- Mixed Integer Programming (MIP)
- Balances construction costs and transportation costs
- Capacity planning with demand fulfillment

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| UI Framework | Streamlit |
| Optimization | PuLP (Linear Programming) |
| Graph Algorithm | NetworkX |
| Visualization | Matplotlib |
| Data Processing | Pandas, NumPy |

---

## 🚀 Live Demo

**🌐 Web App:** https://logistics-app-demo-lun9fabpdgmefrpuubttuf.streamlit.app/

---

## 📁 Project Structure

```
logistics-app-demo/
├── ap.py              # Main Streamlit application
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── scripts/           # Automation scripts
├── .devcontainer/     # VS Code Dev Container config
└── .github/           # GitHub Actions
```

---

## 🔧 Features

- **Bilingual Interface**: Switch between Chinese and English
- **Interactive Visualization**: Route maps and cost breakdown charts
- **Real-time Optimization**: Instant LP/MIP solution feedback
- **GitHub Actions Integration**: Automated CI/CD workflow

---

## 📝 License

MIT License - Feel free to use and modify.

---

**Built with Operations Research & Python** 🐍