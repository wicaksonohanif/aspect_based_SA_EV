/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(90deg, #E81A1D 0%, #F7947E 100%);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-size: 15px;
    font-weight: 600;
}
            
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    padding: 10px 15px;
    border-radius: 25px;
    border: 2px solid transparent; 
    margin-bottom: 5px;
    transition: all 0.3s ease;
    cursor: pointer;
    width: 100%;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    border: 2px solid #FFFFFF !important;
    background-color: rgba(255, 255, 255, 0.15) !important;
}
            
.sidebar-logo-card {
    background-color: #FFFFFF;
    border-radius: 20px;
    padding: 10px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
}
.sidebar-logo-card h2 {
    color: #E81A1D !important;
    margin: 0;
    font-size: 25px;
    font-weight: 800;
}