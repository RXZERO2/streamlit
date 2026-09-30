import streamlit as st
st.title('My First Streamlit App')
st.write('Hello, Streamlit!')
name = st.text_input('Enter your name:')
if name:
    st.success(f'Welcome, {name}!')

import streamlit as st
import math

st.set_page_config(page_title="Streamlit Calculator", page_icon="🧮", layout="centered")

# Custom CSS for calculator styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Display screen styling */
    .calc-display {
        background-color: #1e293b;
        color: #38bdf8;
        font-family: 'Courier New', Courier, monospace;
        font-size: 2.2rem;
        font-weight: bold;
        text-align: right;
        padding: 15px 20px;
        border-radius: 12px;
        border: 2px solid #334155;
        margin-bottom: 20px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
        word-break: break-all;
        min-height: 70px;
    }

    /* Sub-display for expression history context */
    .calc-subdisplay {
        color: #94a3b8;
        font-size: 0.9rem;
        text-align: right;
        margin-bottom: -15px;
        margin-top: -10px;
        min-height: 20px;
    }

    /* Button hover & active feedback */
    div.stButton > button {
        width: 100%;
        height: 60px;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        transition: all 0.15s ease !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    div.stButton > button:active {
        transform: translateY(0);
    }
</style>
""", unsafe_allow_html=True)

if "expression" not in st.session_state:
    st.session_state.expression = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
if "history" not in st.session_state:
    st.session_state.history = []

def append_char(char):
    """Appends a character or number to the display expression."""
    st.session_state.expression += str(char)

def clear_all():
    """Clears current expression and reset subdisplay."""
    st.session_state.expression = ""
    st.session_state.last_result = ""

def backspace():
    """Removes the last character from the expression."""
    st.session_state.expression = st.session_state.expression[:-1]

def calculate():
    """Evaluates the expression safely and logs history."""
    expr = st.session_state.expression
    if not expr:
        return
    
    try:
        # Format string for evaluation (replacing display symbols with Python math operators)
        safe_expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
        
        # Evaluate arithmetic expression
        result = eval(safe_expr, {"__builtins__": None, "sqrt": math.sqrt, "math": math})
        
        # Round floating numbers if necessary for cleaner output
        if isinstance(result, float):
            result = round(result, 8)
            if result.is_integer():
                result = int(result)
                
        res_str = str(result)
        st.session_state.history.append(f"{expr} = {res_str}")
        st.session_state.last_result = f"{expr} ="
        st.session_state.expression = res_str
    except Exception:
        st.session_state.last_result = "Error"
        st.session_state.expression = ""

def apply_sqrt():
    """Calculates square root of current number or expression."""
    expr = st.session_state.expression
    if not expr:
        return
    try:
        safe_expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
        val = eval(safe_expr, {"__builtins__": None, "sqrt": math.sqrt, "math": math})
        if val < 0:
            st.session_state.last_result = "Invalid Input (sqrt < 0)"
            st.session_state.expression = ""
        else:
            res = math.sqrt(val)
            if res.is_integer():
                res = int(res)
            else:
                res = round(res, 8)
            res_str = str(res)
            st.session_state.history.append(f"√({expr}) = {res_str}")
            st.session_state.last_result = f"√({expr}) ="
            st.session_state.expression = res_str
    except Exception:
        st.session_state.last_result = "Error"
        st.session_state.expression = ""

st.title("🧮 Interactive Calculator")

# Calculator Main Display Screen
st.markdown(f'<div class="calc-subdisplay">{st.session_state.last_result}</div>', unsafe_allow_html=True)
display_text = st.session_state.expression if st.session_state.expression != "" else "0"
st.markdown(f'<div class="calc-display">{display_text}</div>', unsafe_allow_html=True)

# Row 1: Function buttons (Clear, Backspace, Square Root, Exponent)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button("AC", on_click=clear_all, type="primary", use_container_width=True)
with col2:
    st.button("⌫", on_click=backspace, use_container_width=True)
with col3:
    st.button("√", on_click=apply_sqrt, use_container_width=True)
with col4:
    st.button("^", on_click=append_char, args=("^",), use_container_width=True)

# Row 2: Numbers 7-9 and Division
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button("7", on_click=append_char, args=("7",), use_container_width=True)
with col2:
    st.button("8", on_click=append_char, args=("8",), use_container_width=True)
with col3:
    st.button("9", on_click=append_char, args=("9",), use_container_width=True)
with col4:
    st.button("÷", on_click=append_char, args=("÷",), use_container_width=True)

# Row 3: Numbers 4-6 and Multiplication
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button("4", on_click=append_char, args=("4",), use_container_width=True)
with col2:
    st.button("5", on_click=append_char, args=("5",), use_container_width=True)
with col3:
    st.button("6", on_click=append_char, args=("6",), use_container_width=True)
with col4:
    st.button("×", on_click=append_char, args=("×",), use_container_width=True)

# Row 4: Numbers 1-3 and Subtraction
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button("1", on_click=append_char, args=("1",), use_container_width=True)
with col2:
    st.button("2", on_click=append_char, args=("2",), use_container_width=True)
with col3:
    st.button("3", on_click=append_char, args=("3",), use_container_width=True)
with col4:
    st.button("-", on_click=append_char, args=("-",), use_container_width=True)

# Row 5: Decimal, 0, Equals, and Addition
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button(".", on_click=append_char, args=(".",), use_container_width=True)
with col2:
    st.button("0", on_click=append_char, args=("0",), use_container_width=True)
with col3:
    st.button("=", on_click=calculate, type="primary", use_container_width=True)
with col4:
    st.button("+", on_click=append_char, args=("+",), use_container_width=True)

st.divider()
with st.expander("📋 Calculation History", expanded=False):
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-10:]), 1):
            st.text(f"{idx}. {item}")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("No calculations done yet.")

col1, col2 = st.columns(2)
with col1:
    st.write('Left')
with col2:
    st.write('Right')

# Sidebar
st.sidebar.title('Menu')
# Tabs
tab1, tab2 = st.tabs(['A',
'B'])
with tab1:
    st.write('Tab A')