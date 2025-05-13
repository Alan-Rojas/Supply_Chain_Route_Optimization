# This is a rework of a past project, trying to make it cleanner and neater. 
import random
import numpy as np
import pandas as pd
import folium
from sklearn.cluster import KMeans
from folium import plugins
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from sklearn.metrics.pairwise import haversine_distances
from math import radians
#-------------------------------------Data Preparation-----------------------------------------
def data_cleanning():
    """
    Fixes formating errors inside the data file. 
    """

    stores = pd.read_excel('edo_mex_ComeVerde.xlsx')
    stores.rename(columns = {"Unnamed: 0": "id", "Latitud": "lat", "Longitud": "lon", "Frecuencia": "freq"}, inplace = True)
    return stores

def clustering(df, k = 7):
  """
  Applies Kmeans algorithm to create clusters within stores, making the promotor selection easier. 
  """

  coords = df[['lat', 'lon']]
  kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
  df['cluster'] = kmeans.fit_predict(coords)

  
  #print(df['cluster'].value_counts().sort_index())
  return df

def show_clusters_on_map(df):

    """
    Creates a map to visualize all stores. 
    """
    # Map is created in the center of all the stores
    cent_lat = df["lat"].mean()
    cent_lon = df["lon"].mean()
    map = folium.Map(location=[cent_lat, cent_lon], zoom_start=12)

    # Different colors for each cluster. 
    n_clusters = df["cluster"].nunique()
    palette = sns.color_palette("hls", n_clusters).as_hex()

    for _, store in df.iterrows():
        cluster_id = store["cluster"]
        folium.CircleMarker(
            location=[store["lat"], store["lon"]],
            radius=5,
            color=palette[cluster_id],
            fill=True,
            fill_opacity=0.7,
            tooltip=f"ID {store['id']} | Cluster {cluster_id}"
        ).add_to(map)

    # Agregar una capa de agrupamiento opcional (puede quitarse si quieres solo los clusters)
    plugins.MarkerCluster().add_to(map)

    return map

def drop_nonsense_stores(stores):
    """
    Ok so from here we can see that some stores do not make sense, hence, we will drop them by id.

    Id's to drop: 125, 56, 102, 209, 58, 310, 352,
    """

    ids_to_drop = [125, 56, 102, 209, 58, 310, 352]
    stores = stores[~stores['id'].isin(ids_to_drop)]
    stores = stores[stores['cluster'] != 1]
    stores = clustering(stores)
    return stores

def create_clusters(stores):

    """
    Creates a dictionary with the stores according to their cluster. 
    """
    clusters = {}
    for i in range(len(stores['cluster'].unique())):
        clusters[i] = stores[stores['cluster'] == i]
    return clusters

stores = data_cleanning()
stores = clustering(stores)
stores = drop_nonsense_stores(stores)
clusters = create_clusters(stores)

# ---------------------------------------Genetic Algorithm ---------------------------------------#

def generate_initial_pop(cluster_i_df, n_ind=50, route_size=6):

    """
    Generates the initial population for a cluster (a promotor).

    Params:
    - cluster_i_df,
    - n_ind, 
    - route_size, 
    Returns:
    - pob: List of individuals, where each is a set of routes for the promotor.
    """
    pob = []

    for _ in range(n_ind):

        visits = []

        # Each store has ti be visited freq times, why don't create freq stores. 
        for _, tienda in cluster_i_df.iterrows():
            visits.extend([tienda.to_dict()] * int(tienda['freq']))

        random.shuffle(visits)

        total_visits = len(visits)
        n_routes = total_visits // route_size
        rest = total_visits % route_size

        routes = []
        idx = 0
        for _ in range(n_routes):
            routes.append(visits[idx:idx + route_size])
            idx += route_size

        # Last route will have the rest of stores that remained unassigned to a route.
        if rest > 0:
            routes.append(visits[idx:])   

        pob.append(routes)

    return pob

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the distance in kilometers between two points given by latitude and longitude.
    """
    earth_radius_km = 6371  # Earth's radius in kilometers according to google

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = earth_radius_km * c
    return distance

def calculate_total_distance(route):

    """
    Calculates the total distance in kilometers of a route given a list of dictionaries
    with 'lat' and 'lon'.
    """
    total_distance_km = 0
    for i in range(len(route) - 1):
        lat_start = route[i]['lat']
        lon_start = route[i]['lon']
        lat_end = route[i + 1]['lat']
        lon_end = route[i + 1]['lon']
        total_distance_km += haversine(lat_start, lon_start, lat_end, lon_end)
    return total_distance_km

def evaluate_fitness(plan, cluster_df):
    """
    Evaluates the fitness of a complete plan (multiple routes).
    Rewards short distances and penalizes if store frequencies are not respected.

    plan: List of routes, where each route is a list of dicts (stores)
    cluster_df: Original DataFrame of the cluster to verify correct frequencies
    """
    # --- Total distance of the plan
    total_distance = sum([calculate_total_distance(route) for route in plan])

    # --- Penalty for frequency error
    visits_made = Counter()
    for route in plan:
        for store in route:
            visits_made[store['id']] += 1

    expected_visits = dict(zip(cluster_df['id'], cluster_df['freq']))
    total_error = 0
    for store_id, expected_freq in expected_visits.items():
        error = abs(visits_made.get(store_id, 0) - expected_freq)
        total_error += error

    # --- Final fitness: penalty + distance
    fitness = total_distance + (total_error * 1000)  # strongly penalizes if frequencies are not respected

    return fitness

def order_by_nearest_neighbor(route):
    """
    Orders a list of stores using the nearest neighbor algorithm.
    The input and output are lists of dicts with keys like 'id', 'lat', 'lon', etc.
    """
    if not route:
        return []

    unvisited = route.copy()
    ordered = [unvisited.pop(0)]  # Starts at the first store (can be randomized if desired)

    while unvisited:
        current = ordered[-1]
        next_store = min(
            unvisited,
            key=lambda store: np.linalg.norm(
                np.array([current['lat'], current['lon']]) - np.array([store['lat'], store['lon']])
            )
        )
        ordered.append(next_store)
        unvisited.remove(next_store)

    return ordered

def tournament_selection(population, scores, tournament_size=3):
    """
    Performs tournament selection.
    From k randomly selected individuals, chooses the one with the lowest score (best fitness).

    Args:
        population: list of plans (each plan is a list of routes).
        scores: list of scores corresponding to each plan.
        tournament_size: number of competitors in each tournament.

    Returns:
        A plan (individual) that won the tournament.
    """
    competitors = random.sample(list(zip(population, scores)), tournament_size)
    winner = min(competitors, key=lambda item: item[1])  # Lower score = better fitness
    return winner[0]

def crossover_plans(parent1, parent2, probability=0.8):
    """
    Crossovers two parents (plans) with a certain probability and a random cut point.

    Args:
        parent1: plan (list of routes).
        parent2: plan (list of routes).
        probability: probability of crossing the parents.

    Returns:
        Two children (new plans).
    """
    if random.random() > probability:
        return parent1[:], parent2[:]

    # Random cut point
    cut_point = random.randint(1, min(len(parent1), len(parent2)) - 1)

    child1 = parent1[:cut_point] + parent2[cut_point:]
    child2 = parent2[:cut_point] + parent1[cut_point:]

    return child1, child2

def mutate_plan(plan, probability=0.2):
    """
    Applies mutation to a plan (list of routes), swapping two stores in a random route.

    Args:
        plan: list of routes (each route is a list of stores).
        probability: probability of mutating the plan.

    Returns:
        The possibly mutated plan.
    """
    mutated_plan = plan[:]

    if random.random() < probability:
        # Select a random route from the plan
        route_index = random.randint(0, len(mutated_plan) - 1)
        route = mutated_plan[route_index][:]

        if len(route) >= 2:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]

        mutated_plan[route_index] = route

    return mutated_plan

def generate_next_generation(current_population, stores_df, population_size):
    next_generation = []
    while len(next_generation) < population_size:
        parent1 = tournament_selection(current_population, stores_df)
        parent2 = tournament_selection(current_population, stores_df)
        child1, child2 = crossover_plans(parent1, parent2)

        mutated_child1 = mutate_plan(child1)
        mutated_child2 = mutate_plan(child2)

        # Order routes by nearest neighbor (TSP-like)
        ordered_child1 = [order_by_nearest_neighbor(route) for route in mutated_child1]
        ordered_child2 = [order_by_nearest_neighbor(route) for route in mutated_child2]

        next_generation.append(ordered_child1)
        if len(next_generation) < population_size:
            next_generation.append(ordered_child2)
            
    return next_generation

def evolve_cluster(
    cluster_df,
    generations=100,
    population_size=50,
    crossover_rate=0.8,
    mutation_rate=0.2,
    tournament_k=5
):
    """
    Evolves a cluster of stores to find an optimized routing plan using a genetic algorithm.

    Args:
        cluster_df: DataFrame of the cluster with store information (id, lat, lon, freq).
        generations: Number of generations to run the genetic algorithm.
        population_size: Number of individuals (plans) in the population.
        crossover_rate: Probability of crossover between two parent plans.
        mutation_rate: Probability of mutation within a plan.
        tournament_k: Number of individuals participating in each tournament selection.

    Returns:
        The best routing plan found (list of routes, each route is a list of stores).
    """
    # Generate initial population
    population = generate_initial_pop(cluster_df, population_size)

    best_fitness = float('inf')
    best_plan = None

    for generation in range(generations):
        new_population = []
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, cluster_df, tournament_size=tournament_k)
            parent2 = tournament_selection(population, cluster_df, tournament_size=tournament_k)

            # Crossover
            children = crossover_plans(parent1, parent2, probability=crossover_rate)

            # Mutation
            children = [mutate_plan(child, probability=mutation_rate) for child in children]

            new_population.extend(children)

        # Maintain population of the same size
        population = new_population[:population_size]

        # Evaluate the best plan of this generation
        generation_fitness = [(evaluate_fitness(plan, cluster_df), plan) for plan in population]
        generation_fitness.sort(key=lambda x: x[0])

        if generation_fitness[0][0] < best_fitness:
            best_fitness = generation_fitness[0][0]
            best_plan = generation_fitness[0][1]
        # print(f"Generation: {generation}")

    # Order final routes with nearest neighbor for tidiness
    best_plan_ordered = [order_by_nearest_neighbor(route) for route in best_plan]

    return best_plan_ordered

def show_route_on_map(route, color='blue'):
    """
    Displays a route on a map using Folium.

    Args:
        route: A list of dictionaries, where each dictionary represents a store
               with 'lat', 'lon', 'Formato' (format), and 'freq' (frequency).
        color: The color of the route line on the map.

    Returns:
        A Folium map object.
    """
    if not route:
        print("Empty route")
        return

    # Create map centered on the first store
    center = [route[0]['lat'], route[0]['lon']]
    map_object = folium.Map(location=center, zoom_start=12)

    # Mark stores and draw lines
    coordinates = []
    for i, store in enumerate(route):
        coord = [store['lat'], store['lon']]
        coordinates.append(coord)
        folium.Marker(
            location=coord,
            popup=f"{store['store']}",
            tooltip=f"{i+1}. {store['store']}"
        ).add_to(map_object)

    # Draw route line
    folium.PolyLine(locations=coordinates, color=color, weight=3).add_to(map_object)

    # Calculate total distance
    total_distance = calculate_total_distance(route)

    #print(f"Total distance: {total_distance:.2f} km")
    return map_object, total_distance

def best_routes(clusters):
    """
    Finds the best routing plan for each cluster.

    Args:
        clusters: A dictionary where keys are cluster IDs and values are DataFrames
                  for each cluster.

    Returns:
        A dictionary where keys are cluster IDs and values are the best routing
        plan found for that cluster.
    """
    best_routes_per_cluster = {}
    for cluster_id, cluster_df in clusters.items():
        # print(f"Cluster {cluster_id}")
        best_plan = evolve_cluster(cluster_df)
        best_routes_per_cluster[cluster_id] = best_plan
    return best_routes_per_cluster

"""
So here is the deal, each promotor must have 24 routes, so they work 6 days a week. 
Promotors that have more than 24 routes will have some of theirs passed to another promotor. 
At the end of the process, no promotor should have more than 24 routes.
"""

def balance_routes(best_routes, stores_df, max_routes=24):
    """
    Reassigns routes between clusters so that each has a maximum of `max_routes`.

    Parameters:
    - best_routes: dict with routes per cluster (promoter).
    - stores_df: Original DataFrame with lat, lon, and cluster info.
    - max_routes: Maximum number of routes per cluster (default 24).

    Returns:
    - balanced_routes: new dict with reassigned routes.
    """

    # Average coordinates per cluster
    centroids = {}
    for cluster_id in best_routes.keys():
        cluster_stores = stores_df[stores_df['cluster'] == cluster_id]
        centroids[cluster_id] = (
            cluster_stores['lat'].mean(),
            cluster_stores['lon'].mean()
        )

    # Convert to radians for haversine
    centroids_rad = {
        cid: (radians(lat), radians(lon)) for cid, (lat, lon) in centroids.items()
    }

    # Calculate nearest neighbors for each cluster
    neighbors = {}
    for cid in centroids_rad:
        distances = []
        for other_cid in centroids_rad:
            if cid != other_cid:
                d = haversine_distances(
                    [centroids_rad[cid], centroids_rad[other_cid]]
                )[0][1]
                distances.append((other_cid, d))
        neighbors[cid] = [x[0] for x in sorted(distances, key=lambda x: x[1])]

    # Start with a copy of the original routes
    balanced_routes = {cid: routes[:] for cid, routes in best_routes.items()}

    # Review and redistribute excess routes
    for cid in list(balanced_routes.keys()):
        routes = balanced_routes[cid]
        if len(routes) > max_routes:
            excess = routes[max_routes:]
            balanced_routes[cid] = routes[:max_routes]

            # Reassign the excess routes to neighbors
            for route in excess:
                assigned = False
                for neighbor in neighbors[cid]:
                    if len(balanced_routes[neighbor]) < max_routes:
                        balanced_routes[neighbor].append(route)
                        assigned = True
                        break
                if not assigned:
                    print(f"Could not assign a route from cluster {cid} to any neighbor.")

    return balanced_routes

best_routes_ = best_routes(clusters)
balanced_routes = balance_routes(best_routes_, stores)


def export_final_routes_to_csv(final_routes, csv_path="final_routes.csv"):
    """
    Converts the dictionary of final routes into a DataFrame and exports it to a CSV file.

    Parameters:
    - final_routes: dict {promoter_id: [ [route1], [route2], ..., [routeN] ]}
    - csv_path: path where to save the generated CSV.

    Returns:
    - routes_df: Generated DataFrame.
    """
    records = []

    for promoter_id, routes in final_routes.items():
        for route_index, route in enumerate(routes):
            for visit_order, store in enumerate(route):
                records.append({
                    "promoter": promoter_id,
                    "route_id": route_index,
                    "visit_order": visit_order,
                    "store": store.get("Formato", ""),
                    "address": store.get("Address", ""),
                    "lat": store["lat"],
                    "lon": store["lon"]
                })

    routes_df = pd.DataFrame(records)
    routes_df.to_csv(csv_path, index=False)
    return routes_df

final_routes = export_final_routes_to_csv(balanced_routes)