import streamlit as st
import pandas as pd
import numpy as np
import re

# Set pandas option to use the future behavior and avoid deprecation warnings
pd.set_option('future.no_silent_downcasting', True)

def detect_header_row(df, max_rows=30):
    """
    Enhanced header row detection for various quotation formats.
    Uses multiple strategies to identify the most likely header row.
    """
    potential_headers = []
    
    # Check only the first several rows to find headers
    check_df = df.iloc[:max_rows].copy()
    
    # Convert everything to string for consistent analysis
    for col in check_df.columns:
        check_df[col] = check_df[col].astype(str)
    
    # Expanded list of header keywords for different quotation formats
    header_keywords = [
        # Common quotation terms
        'total', 'item', 'description', 'quantity', 'unit', 'rate', 'amount', 
        'price', 'no.', 'no', 'code', 'ref', 'date', 'cost',
        # Additional keywords for different formats
        'qty', 'uom', 'part', 'sku', 'article', 'product', 'service',
        'net', 'gross', 'vat', 'tax', 'discount', 'subtotal',
        'line', 'pos', 'position', 'quote', 'offer', 'bid', 'proposal',
        'spec', 'specification', 'detail', 'particulars',
        # Construction specific
        'boq', 'material', 'labor', 'labour', 'markup', 'work',
        # IT/Tech specific
        'license', 'subscription', 'per user', 'implementation',
        # Manufacturing specific
        'part number', 'bom', 'tooling', 'setup'
    ]
    
    for i in range(min(max_rows, len(check_df))):
        row = check_df.iloc[i]
        
        # Skip completely empty rows
        if (row == '').all() or (row == 'nan').all() or (row == 'None').all():
            continue
        
        # Multiple strategies to score potential header rows
        
        # Strategy 1: Non-numeric text content (headers tend to be text)
        non_numeric = sum(1 for val in row if not str(val).replace('.', '', 1).isdigit() 
                          and val != '' and val.lower() != 'nan' and val.lower() != 'none')
        
        # Strategy 2: Specific keywords often found in headers
        keyword_matches = sum(2 for val in row if any(keyword.lower() in str(val).lower() 
                                                     for keyword in header_keywords))
        
        # Strategy 3: Capitalization patterns (common in headers)
        capitalized = sum(1 for val in row if str(val).istitle() or str(val).isupper())
        
        # Strategy 4: Row followed by numeric data (headers often precede numeric data)
        followed_by_numeric = 0
        if i < len(check_df) - 1:
            next_row = check_df.iloc[i+1]
            numeric_cells_in_next = sum(1 for val in next_row 
                                       if val.replace('.', '', 1).replace(',', '').isdigit())
            if numeric_cells_in_next >= 2:  # At least a few numeric cells suggest data rows
                followed_by_numeric = 3
        
        # Strategy 5: Column-like structure (headers typically span multiple columns)
        column_structure = 0
        if non_numeric >= 3 and row.astype(str).str.strip().str.len().mean() < 20:
            column_structure = 2
        
        # Calculate a combined score based on these heuristics
        score = non_numeric + keyword_matches + capitalized + followed_by_numeric + column_structure
        
        potential_headers.append((i, score))
    
    # Sort by score in descending order
    potential_headers.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top candidate row index
    return potential_headers[0][0] if potential_headers else 0

def is_data_row(row, header_col_count=None):
    """
    Enhanced data row detection for various quotation formats.
    More robust identification of relevant data rows.
    """
    # Convert to series if it's not already
    if not isinstance(row, pd.Series):
        row = pd.Series(row)
    
    # Skip empty rows
    # Use a cleaner approach to check for empty rows
    is_empty = True
    for val in row:
        if pd.notna(val) and str(val).strip() not in ('', 'nan', 'None'):
            is_empty = False
            break
    if is_empty:
        return False
    
    # Convert to string for analysis
    row_str = row.astype(str)
    
    # Check if first cell is numeric or item-code-like
    first_cell = str(row.iloc[0]).strip().lower()
    
    # More flexible patterns for item numbers or codes
    first_cell_is_item_number = any([
        # Pure numbers (1, 2, 3)
        first_cell.isdigit(),
        # Alphanumeric item codes (1a, 2b, etc.)
        re.match(r'^\d+[a-zA-Z]?$', first_cell) is not None,
        # Item codes with periods or dashes (1.1, 2-3, etc.)
        re.match(r'^\d+[\.\-]\d+$', first_cell) is not None,
        # SKU-like codes
        re.match(r'^[a-zA-Z]\d+$', first_cell) is not None
    ])
    
    # Check for section headers (like "A", "B", "GENERAL CONSTRUCTION")
    is_section_header = any([
        # Single letters that are section markers
        (len(first_cell) == 1 and first_cell.isalpha()),
        # Roman numerals that might be section markers
        re.match(r'^[ivxIVX]+$', first_cell) is not None,
        # ALL CAPS description that's likely a section title
        (len(first_cell) > 2 and first_cell.isupper())
    ])
    
    # Expanded check for footer/total rows
    total_keywords = ['total', 'subtotal', 'sub-total', 'grand total', 'sub total', 
                     'sum', 'amount', 'net', 'gross', 'final']
    is_total_row = any(keyword in row_str.str.lower().str.strip().values for keyword in total_keywords)
    
    # Enhanced check for reference/note rows
    note_keywords = ['refer', 'note', 'see', 'reference', 'detail', 'spec']
    is_reference_row = (
        (pd.isna(row.iloc[0]) or str(row.iloc[0]).strip() == "") and 
        any(keyword in str(val).lower() for val in row for keyword in note_keywords if not pd.isna(val))
    )
    
    # Check for numeric values in any potential amount columns
    amount_cols = []
    
    # Different quotation formats might have amount columns in different positions
    # Check all columns except first (typically item/code) and second (typically description)
    for i in range(2, min(len(row), 8)):  # Check a reasonable number of columns
        val = row.iloc[i]
        if pd.notna(val) and (
            isinstance(val, (int, float)) or 
            (isinstance(val, str) and re.match(r'^[\d,]+(\.\d+)?$', val.strip().replace(',', '')))
        ):
            amount_cols.append(i)
    
    has_amount = len(amount_cols) > 0
    
    # Main logic for identifying data rows
    is_data = (
        (first_cell_is_item_number or is_section_header) and
        not is_total_row and
        not is_reference_row and
        has_amount
    )
    
    return is_data

def extract_relevant_data(df, header_row_idx):
    """
    Extract all relevant data rows based on the header row.
    Now handles various quotation formats better.
    """
    # Get the cleaned and properly named header
    header = df.iloc[header_row_idx]
    
    # Count non-empty columns in header for reference
    header_col_count = sum(1 for val in header if pd.notna(val) and str(val).strip() != '')
    
    # Create a mask for data rows
    data_rows_mask = df.iloc[(header_row_idx+1):].apply(
        lambda row: is_data_row(row, header_col_count), 
        axis=1
    )
    
    # Get indices of data rows
    data_row_indices = np.where(data_rows_mask)[0] + header_row_idx + 1
    
    # Include section headers as well (rows with A, B, C, etc. for better organization)
    section_headers = []
    for i in range(header_row_idx+1, df.shape[0]):
        first_cell = str(df.iloc[i, 0]).strip()
        # More comprehensive section header detection
        if ((len(first_cell) == 1 and first_cell.isalpha() and first_cell.isupper()) or
            re.match(r'^[IVX]+\.?$', first_cell) is not None or  # Roman numerals
            (first_cell.isupper() and len(first_cell.split()) <= 3)):  # SHORT ALL-CAPS TITLES
            section_headers.append(i)
    
    # Also include critical rows that might contain totals, subtotals, etc.
    total_rows = []
    for i in range(header_row_idx+1, df.shape[0]):
        row_text = ' '.join(str(x).lower() for x in df.iloc[i] if pd.notna(x))
        if any(keyword in row_text for keyword in ['total', 'subtotal', 'grand total']):
            # Only include if it also has numeric values (likely a total amount)
            numeric_values = sum(1 for val in df.iloc[i] if 
                              pd.notna(val) and (
                                  isinstance(val, (int, float)) or 
                                  (isinstance(val, str) and 
                                   re.match(r'^[\d,]+(\.\d+)?$', str(val).strip().replace(',', '')))
                              ))
            if numeric_values > 0:
                total_rows.append(i)
    
    # Combine header row with data rows, section headers, and totals
    result_indices = [header_row_idx] + section_headers + data_row_indices.tolist() + total_rows
    result_indices = sorted(set(result_indices))  # Remove duplicates and sort
    
    # Return the filtered dataframe
    return df.iloc[result_indices].reset_index(drop=True)

def create_unique_column_names(header_row):
    """Create unique column names from the header row values."""
    unique_columns = []
    column_counts = {}
    
    for col_val in header_row:
        col_name = str(col_val).strip()
        if pd.isna(col_val) or col_name == '' or col_name.lower() == 'nan' or col_name.lower() == 'none':
            col_name = f'Unnamed_{len(unique_columns)}'
        
        if col_name in column_counts:
            column_counts[col_name] += 1
            unique_columns.append(f"{col_name}_{column_counts[col_name]}")
        else:
            column_counts[col_name] = 0
            unique_columns.append(col_name)
    
    return unique_columns

def clean_data_for_display(df):
    """Safely convert dataframe values to strings for display."""
    display_df = df.copy()
    
    for col in display_df.columns:
        try:
            display_df[col] = display_df[col].astype(str)
        except Exception:
            # If conversion fails, try cell-by-cell
            for idx in display_df.index:
                try:
                    display_df.at[idx, col] = str(display_df.at[idx, col])
                except:
                    display_df.at[idx, col] = "Error converting value"
    
    return display_df

def process_sheet(df, sheet_name):
    """Process a single sheet and return results."""
    results = {}
    
    try:
        # Detect header row
        header_row = detect_header_row(df)
        results['header_row'] = header_row
        
        # Extract relevant data
        relevant_data = extract_relevant_data(df, header_row)
        results['relevant_data'] = relevant_data
        
        # Set the first row as the header, but handle duplicate column names
        data_with_header = relevant_data.copy()
        
        # Extract header row and make a list of column names
        header_row_values = data_with_header.iloc[0].astype(str)
        
        # Create unique column names
        unique_columns = create_unique_column_names(header_row_values)
        
        # Assign the unique column names to the dataframe
        data_with_header.columns = unique_columns
        data_with_header = data_with_header.iloc[1:].reset_index(drop=True)
        
        results['data_with_header'] = data_with_header
        results['success'] = True
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
    
    return results

def main():
    st.title("Multi-Sheet Excel File Header and Data Extractor")
    
    # Add an explanation about what the tool does
    st.markdown("""
    This tool extracts relevant data rows from Excel files, especially quotations and estimates.
    It automatically:
    1. Processes each sheet in the Excel file separately
    2. Detects the header row for each sheet
    3. Identifies actual data rows (items with numbers, descriptions, and amounts)
    4. Filters out reference notes and irrelevant content
    5. Creates a clean dataset with uniquely named columns
    """)
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        
        try:
            # Get list of sheet names
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            st.write(f"**Found {len(sheet_names)} sheets:** {', '.join(sheet_names)}")
            
            # Create tabs for each sheet
            tabs = st.tabs(sheet_names)
            
            # Process each sheet
            for i, sheet_name in enumerate(sheet_names):
                with tabs[i]:
                    st.subheader(f"Sheet: {sheet_name}")
                    
                    # Read the sheet without headers
                    df_no_header = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
                    
                    # Process the sheet
                    sheet_results = process_sheet(df_no_header, sheet_name)
                    
                    if sheet_results['success']:
                        header_row = sheet_results['header_row']
                        relevant_data = sheet_results['relevant_data']
                        data_with_header = sheet_results['data_with_header']
                        
                        # Display the detected header row with preview
                        st.write(f"**Detected header row:** Row {header_row+1}")
                        
                        # Allow user to confirm or choose a different header row
                        with st.expander("Review and adjust header row if needed"):
                            st.dataframe(clean_data_for_display(df_no_header.iloc[max(0, header_row-2):header_row+3]))
                            
                            custom_header = st.number_input(
                                f"Adjust header row for {sheet_name} if needed:", 
                                min_value=1, 
                                max_value=min(50, len(df_no_header)), 
                                value=header_row+1,
                                key=f"header_input_{sheet_name}"
                            ) - 1
                            
                            if custom_header != header_row:
                                # Reprocess with new header row
                                header_row = custom_header
                                st.write(f"Using row {header_row+1} as header")
                                
                                # Extract relevant data with new header
                                relevant_data = extract_relevant_data(df_no_header, header_row)
                                
                                # Set the first row as the header
                                data_with_header = relevant_data.copy()
                                header_row_values = data_with_header.iloc[0].astype(str)
                                unique_columns = create_unique_column_names(header_row_values)
                                data_with_header.columns = unique_columns
                                data_with_header = data_with_header.iloc[1:].reset_index(drop=True)
                        
                        # Show basic info about the extracted data
                        st.write(f"**Original Rows:** {df_no_header.shape[0]}, **Extracted Rows:** {relevant_data.shape[0]}")
                        
                        # Show column names
                        st.write("**Detected Columns:**")
                        st.write(", ".join(str(col) for col in data_with_header.columns.tolist()))
                        
                        # Display data preview with the detected header
                        st.write("**Data Preview with Detected Header:**")
                        # Show more rows in the preview
                        display_df = clean_data_for_display(data_with_header.head(10))
                        st.dataframe(display_df)
                        
                        # Allow the user to download the extracted data for this sheet
                        csv = data_with_header.to_csv(index=False)
                        st.download_button(
                            label=f"Download extracted data for {sheet_name} as CSV",
                            data=csv,
                            file_name=f"{sheet_name}_extracted_data.csv",
                            mime="text/csv",
                            key=f"download_{sheet_name}"
                        )
                        
                        # Show original data for comparison
                        with st.expander("View Original Data Around Detected Header"):
                            # Convert all columns to string to ensure Arrow compatibility
                            preview_df = clean_data_for_display(df_no_header.iloc[max(0, header_row-3):header_row+10])
                            st.dataframe(preview_df)
                    else:
                        st.error(f"Error processing sheet '{sheet_name}': {sheet_results['error']}")
            
            # Option to download all sheets as a combined ZIP file
            if len(sheet_names) > 1:
                st.write("---")
                st.info("You can download each sheet individually using the buttons above.")
                
        except Exception as e:
            st.error(f"Error processing the file: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()