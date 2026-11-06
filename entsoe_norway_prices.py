"""
Script to fetch hourly energy prices for Norwegian bidding zones from ENTSOE Transparency Platform
Date range: December 1, 2020 to March 26, 2021
"""

import requests
from datetime import datetime, timedelta
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict
import time

# ENTSOE API configuration
BASE_URL = "https://web-api.tp.entsoe.eu/api"

# Norwegian bidding zones (EIC codes)
NORWEGIAN_ZONES = {
    'NO1': '10YNO-1--------2',  # Oslo
    'NO2': '10YNO-2--------T',  # Kristiansand
    'NO3': '10YNO-3--------J',  # Trondheim
    'NO4': '10YNO-4--------9',  # Tromsø
    'NO5': '10Y1001A1001A48H'   # Bergen
}

#------------------------------------------------------------------------------------------------
# Get Security Token
#------------------------------------------------------------------------------------------------
def get_security_token(email: str, password: str) -> str:
    """
    Authenticate with ENTSOE and retrieve API security token.
    
    Note: The ENTSOE Transparency Platform typically uses API tokens rather than 
    email/password authentication for API access. You may need to:
    1. Log in to https://transparency.entsoe.eu/
    2. Go to your account settings
    3. Generate an API key/security token
    
    For now, this function returns a placeholder. Replace with your actual API token.
    """
    # TODO: Replace this with your actual ENTSOE API token
    # You can get this from: https://transparency.entsoe.eu/ -> Account Settings -> Web API Security Token
    print("Note: ENTSOE API requires a security token from your account settings.")
    print("Please obtain your token from: https://transparency.entsoe.eu/ -> Settings -> Web API Security Token")
    
    # Placeholder - replace with your actual token
    return "YOUR_API_TOKEN_HERE"


#------------------------------------------------------------------------------------------------
# Fetch Day-Ahead Prices
#------------------------------------------------------------------------------------------------
def fetch_day_ahead_prices(zone_code: str, start_date: datetime, end_date: datetime, 
                           security_token: str) -> pd.DataFrame:
    """
    Fetch day-ahead prices for a specific bidding zone and date range.
    
    Args:
        zone_code: EIC code for the bidding zone
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        security_token: ENTSOE API security token
    
    Returns:
        DataFrame with timestamp and price columns
    """
    # Format dates for API (YYYYMMDDHHMM)
    start_str = start_date.strftime('%Y%m%d%H%M')
    end_str = end_date.strftime('%Y%m%d%H%M')
    
    # API parameters for day-ahead prices
    params = {
        'securityToken': security_token,
        'documentType': 'A44',  # Price Document
        'in_Domain': zone_code,
        'out_Domain': zone_code,
        'periodStart': start_str,
        'periodEnd': end_str
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Extract price data from XML
        prices = []
        timestamps = []
        
        # Define namespace
        ns = {'ns': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3'}
        
        for timeseries in root.findall('.//ns:TimeSeries', ns):
            for period in timeseries.findall('.//ns:Period', ns):
                # Get period start time
                period_start = period.find('.//ns:timeInterval/ns:start', ns).text
                period_start_dt = datetime.strptime(period_start, '%Y-%m-%dT%H:%MZ')
                
                # Get all points in the period
                for point in period.findall('.//ns:Point', ns):
                    position = int(point.find('.//ns:position', ns).text)
                    price = float(point.find('.//ns:price.amount', ns).text)
                    
                    # Calculate timestamp (position is 1-indexed, hourly intervals)
                    timestamp = period_start_dt + timedelta(hours=position - 1)
                    
                    timestamps.append(timestamp)
                    prices.append(price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price_eur_mwh': prices
        })
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
    except ET.ParseError as e:
        print(f"Error parsing XML response: {e}")
        return pd.DataFrame()


#------------------------------------------------------------------------------------------------
# Fetch All Norwegian Zones
#------------------------------------------------------------------------------------------------
def fetch_all_norwegian_zones(start_date: datetime, end_date: datetime, 
                               security_token: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch hourly prices for all Norwegian bidding zones.
    
    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        security_token: ENTSOE API security token
    
    Returns:
        Dictionary mapping zone names to DataFrames with price data
    """
    all_data = {}
    
    for zone_name, zone_code in NORWEGIAN_ZONES.items():
        print(f"Fetching data for {zone_name}...")
        
        df = fetch_day_ahead_prices(zone_code, start_date, end_date, security_token)
        
        if not df.empty:
            all_data[zone_name] = df
            print(f"  Retrieved {len(df)} hourly records")
        else:
            print(f"  No data retrieved for {zone_name}")
        
        # Be respectful to the API - add a small delay between requests
        time.sleep(1)
    
    return all_data


#------------------------------------------------------------------------------------------------
# Save Data
#------------------------------------------------------------------------------------------------
def save_data(data_dict: Dict[str, pd.DataFrame], output_format: str = 'csv'):
    """
    Save the fetched data to files.
    
    Args:
        data_dict: Dictionary mapping zone names to DataFrames
        output_format: Output format ('csv' or 'excel')
    """
    if output_format == 'csv':
        # Save each zone to a separate CSV file
        # for zone_name, df in data_dict.items():
        #     filename = f"norway_{zone_name}_prices_2020-12_2021-03.csv"
        #     df.to_csv(filename, index=False)
        #     print(f"Saved {filename}")
        
        # Also create a combined file with all zones
        combined_df = pd.DataFrame()
        for zone_name, df in data_dict.items():
            df_copy = df.copy()
            df_copy['zone'] = zone_name
            combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
        filename = f"norway_all_zones_prices_{combined_df.timestamp.iloc[1].year}"
        combined_df.to_csv(filename + ".csv", index=False)
        print(f"Saved {filename}")
    
    elif output_format == 'excel':
        # Save all zones to different sheets in one Excel file
        with pd.ExcelWriter(filename + ".xlsx") as writer:
            for zone_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=zone_name, index=False)
            
            # Add a combined sheet
            combined_df = pd.DataFrame()
            for zone_name, df in data_dict.items():
                df_copy = df.copy()
                df_copy['zone'] = zone_name
                combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
            combined_df.to_excel(writer, sheet_name='All_Zones', index=False)
        
        print("Saved norway_prices_2020-12_2021-03.xlsx")


#------------------------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------------------------
def main():
    """
    Main function to orchestrate data retrieval.
    """
    # Configuration
    EMAIL = "anf2146@columbia.edu"
    PASSWORD = "w9RuzQ.QCtDvww4"
    
    # Date range
    start_date = datetime(2015, 1, 1, 0, 0)
    end_date = datetime(2015, 12, 31, 23, 59)
    
    print("=" * 60)
    print("ENTSOE Norwegian Energy Price Fetcher")
    print("=" * 60)
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Zones: {', '.join(NORWEGIAN_ZONES.keys())}")
    print()
    
    # Get security token
    print("Important: You need to obtain your API security token from:")
    print("https://transparency.entsoe.eu/ -> Account Settings -> Web API Security Token")
    print()
    
    # You should replace this with your actual token
    security_token = input("Please enter your ENTSOE API security token: ").strip()
    
    if not security_token or security_token == "YOUR_API_TOKEN_HERE":
        print("\nError: Please provide a valid API security token.")
        print("\nTo get your token:")
        print("1. Go to https://transparency.entsoe.eu/")
        print("2. Log in with your credentials")
        print("3. Navigate to Account Settings")
        print("4. Find 'Web API Security Token' and generate/copy it")
        return
    
    # Fetch data for all zones
    print("\nFetching data...")
    all_data = fetch_all_norwegian_zones(start_date, end_date, security_token)
    
    if not all_data:
        print("\nNo data was retrieved. Please check your API token and try again.")
        return
    
    # Save data
    print("\nSaving data...")
    save_data(all_data, output_format='csv')
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for zone_name, df in all_data.items():
        if not df.empty:
            print(f"\n{zone_name}:")
            print(f"  Records: {len(df)}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Price range: {df['price_eur_mwh'].min():.2f} - {df['price_eur_mwh'].max():.2f} EUR/MWh")
            print(f"  Mean price: {df['price_eur_mwh'].mean():.2f} EUR/MWh")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
