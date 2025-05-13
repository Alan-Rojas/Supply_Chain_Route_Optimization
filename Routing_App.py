import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
from Routing import *

def load_routes():
    return pd.read_csv("final_routes.csv")

def show_route_and_dist(promoter, route_id):
    """
    Takes a promoter and a route_id to return a map with the displayed route.
    """
    selected_route = routes[
        (routes["promoter"] == promoter) & 
        (routes["route_id"] == route_id)
    ].sort_values("visit_order")

    route_dicts = selected_route.to_dict(orient="records")  #     
    return show_route_on_map(route_dicts)

def show_info(promoter, route_id):
    """
    Takes a promoter and route_id, and displays information about the route: 
    Promotor, day of that route, the visit order, store and address.
    """
    # Filtrar una sola vez
    df_ruta = routes[(routes["promoter"] == promoter) & (routes["route_id"] == route_id)]
    df_ruta = df_ruta.sort_values("visit_order")  # Ordenar para mostrar en orden de visita

    info = f"**Promotor:** {promoter}\n\n**Day:** {route_id + 1}\n\n**Route:**\n\n"
    for idx, row in df_ruta.iterrows():
        info += f"Visit #{row['visit_order'] + 1}\nStore: {row['store']}\nAddress: {row['address']}\n\n"

    return info

def get_google_maps_link(promoter, route_id):
    """
    Takes a promoter and a route_id and generates a googlemaps link to display the route as an external link.
    """
    df_route = routes[(routes["promoter"] == promoter) & (routes["route_id"] == route_id)]
    df_route = df_route.sort_values("visit_order")

    base = "https://www.google.com/maps/dir/"
    path = "/".join([f"{store['lat']},{store['lon']}" for _, store in df_route.iterrows()])

    return base + path

def main():
    # Streamlit layout
    # ---------------------------
    st.set_page_config(page_title="Promotors Routes", layout="wide")
    st.title("Promotors Routes")

    # Select promoter
    promoters = sorted(routes["promoter"].unique())
    selected_promoter = st.selectbox("Select Promoter", promoters)

    # Select route
    routes_for_promoter = routes[routes["promoter"] == selected_promoter]
    route_ids = sorted(routes_for_promoter["route_id"].unique())
    selected_route = st.selectbox("Select Route (Day)", route_ids)

    # Show button
    show = st.button("**Get Route!**")

    # Info + Map
    if show:
        st.markdown(show_info(selected_promoter, selected_route))

        map = show_route_and_dist(selected_promoter, selected_route)
        link = get_google_maps_link(selected_promoter, selected_route)

        folium_static(map[0])
        st.markdown(f"Total Distance: {map[1]:.2f} km")
        st.markdown(f"[**Start Navigating!**]({link})", unsafe_allow_html = True)

routes = load_routes()
main()
