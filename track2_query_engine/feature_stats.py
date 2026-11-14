import sqlite3

conn = sqlite3.connect('outputs/jetson_tuned/traffic_analytics.sqlite')
cur = conn.cursor()

def fetch(sql):
    return cur.execute(sql).fetchall()

print("=== TABLE ROW COUNTS ===")
print("vehicle_tracks:       ", fetch("SELECT COUNT(*) FROM vehicle_tracks")[0][0])
print("sequences:            ", fetch("SELECT COUNT(*) FROM sequences")[0][0])
print("vehicle_observations: ", fetch("SELECT COUNT(*) FROM vehicle_observations")[0][0])

print("\n=== VEHICLE TYPES ===")
for r in fetch("SELECT vehicle_type, COUNT(*) FROM vehicle_tracks GROUP BY vehicle_type ORDER BY 2 DESC"):
    print(f"  - {r[0]:12s}: {r[1]:4d} ({r[1]/1868*100:.1f}%)")

print("\n=== VEHICLE COLOURS ===")
for r in fetch("SELECT vehicle_colour, COUNT(*) FROM vehicle_tracks GROUP BY vehicle_colour ORDER BY 2 DESC"):
    print(f"  - {r[0]:12s}: {r[1]:4d} ({r[1]/1868*100:.1f}%)")

print("\n=== MOVEMENT DIRECTIONS ===")
for r in fetch("SELECT movement_direction, COUNT(*) FROM vehicle_tracks GROUP BY movement_direction ORDER BY 2 DESC"):
    print(f"  - {r[0]:12s}: {r[1]:4d} ({r[1]/1868*100:.1f}%)")

print("\n=== SPEEDING (LIMIT: 50 KM/H) ===")
for r in fetch("SELECT is_speeding, COUNT(*) FROM vehicle_tracks GROUP BY is_speeding ORDER BY 1 ASC"):
    label = "Normal (<= 50 km/h)" if r[0] == 0 else "Speeding (> 50 km/h)"
    print(f"  - {label:20s}: {r[1]:4d} ({r[1]/1868*100:.1f}%)")

print("\n=== SPEED KINEMATICS (MOVING VEHICLES) ===")
stats = fetch("SELECT MIN(avg_speed_kmh), MAX(avg_speed_kmh), ROUND(AVG(avg_speed_kmh), 1) FROM vehicle_tracks WHERE avg_speed_kmh > 0")[0]
print(f"  - Min Speed: {stats[0]} km/h")
print(f"  - Max Speed: {stats[1]} km/h")
print(f"  - Avg Speed: {stats[2]} km/h")

print("\n=== HOURLY TRAFFIC DISTRIBUTION (TODAY: 2026-11-05) ===")
for r in fetch("SELECT strftime('%H:00', entry_datetime) as hr, COUNT(*) FROM vehicle_tracks GROUP BY hr ORDER BY hr"):
    print(f"  - {r[0]}: {r[1]:3d} vehicles")

print("\n=== PER-SEQUENCE VOLUME ===")
for r in fetch("SELECT sequence_id, COUNT(*) FROM vehicle_tracks GROUP BY sequence_id ORDER BY sequence_id"):
    print(f"  - {r[0]:20s}: {r[1]:3d} tracks")

