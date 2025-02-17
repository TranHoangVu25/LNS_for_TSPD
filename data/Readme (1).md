Data for Vehicle routing problem with drones

Author: Huyen Do Thi Ngoc

Updated time: 17/12/2024 

The file name is made from: "VRPD_Many_visited_Multi_depot_With" + "_Depot_" + NumberOfDepots + "_City_" + NumberOfCities + "_Truck_" + NumberOfTrucks+ "_Drone_" + NumberOfDrones 

For the default value:

NumberOfDepots: 1, 2, 3, 4, 5

NumberOfCities: 100, 200, 300, 400, 500

NumberOfTrucks: 0, 1, 2, 3, 4, 5

NumberOfDrones: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

X, Y, Z: 100, 100, 50  # Dimensions of the area

Customer_Max_Demand: 500

Max_Capacity_Truck: 200

Max_Energy_Truck: 5000

Max_Capacity_Drone: 2

Max_Energy_Drone: 50

Start_Time_Max: 24

End_Time_Max: 48

The file content:

1. In the file, the first line is: "Data for Vehicle routing problem with drones - Author: Huyen Do Thi Ngoc"
2. In the file, the next line is: "Number of depots: " + NumberOfDepots
3. In the file, the next line is: "Number of cities: " + NumberOfCities
4. In the file, the next line is: "Number of trucks: " + NumberOfTrucks
5. In the file, the next line is: "Number of drones: " + NumberOfDrones
6. In the file, the next line is: "Area: " + X + " " + Y + " " + Z
7. In the file, the next line is: "Customer maximum demand: " + Customer_Max_Demand
8. In the file, the next line is: "Max capacity of truck: " + Max_Capacity_Truck
9. In the file, the next line is: "Max energy of truck: " + Max_Energy_Truck
10. In the file, the next line is: "Max capacity of drone: " + Max_Capacity_Drone
11. In the file, the next line is: "Max energy of drone: " + Max_Energy_Drone

12. For next NumberOfDepots lines, each line is made from: Depot_Index + " " + Depot_X + " " + Depot_Y + " " + Dept_Z
  Where:
  Depot_Index is the index of the depot, which count from 1 (for example: 1, 2,.., NumberOfDepots); 
  Depot_X is a random float in the range of 0 and X; 
  Depot_Y is a random float in the range of 0 and Y; 
  Depot_Z is a random float in the range of 0 and Z

13. For next NumberOfCities lines, each line is made from: City_Index + " " + City_X + " " + City_Y + " " + City_Z + " " + Demand + " " + Start_Time + " " + End_Time + " " + Depot_Index 
  Where:
  City_Index is the index of the city, which count from 1 (for example: 1, 2,.., City_Index); 
  City_X is a random float in the range of 0 and X; 
  City_Y is a random float in the range of 0 and Y; 
  City_Z is a random float in the range of 0 and Z; 
  Demand is a random float in the range of 0.1*Customer_Max_Demand and Customer_Max_Demand; 
  Start_Time is a random integer in the range of 0 and Start_Time_Max; 
  End_Time is a random integer in the range of Start_Time and End_Time_Max; 
  Depot_Index is a random integer in the range of 0 and NumberOfDepots (can not take zero value and can take NumberOfDepots value).

14. For next NumberOfTrucks lines, each line is made from: Truck_Index + " " + Capacity_Truck + " " + Energy_Truck + " " + Depot_Index
  Where:
  Truck_Index is the index of the truck, which count from 1 (for example: 1, 2,.., NumberOfTrucks); 
  Capacity_Truck is a random float in the range of 0.5*Max_Capacity_Truck and Max_Capacity_Truck; 
  Energy_Truck is a random float in the range of 0.5*Max_Energy_Truck and Max_Energy_Truck; 
  Depot_Index is a random integer in the range of 0 and NumberOfDepots (can not take zero value and can take NumberOfDepots value) while at least a truck is in at each depot.

15. For next NumberOfDrones lines, each line is made from: Drone_Index + " " + Capacity_Drone + " " + Energy_Drone + " " + Truck_Index
  Where:
  Drone_Index is the index of the drone, which count from 1 (for example: 1, 2,.., NumberOfDrones); 
  Capacity_Drone is a random float in the range of 0.5*Max_Capacity_Drone and Max_Capacity_Drone; 
  Energy_Drone is a random float in the range of 0.5*Max_Energy_Drone and Max_Energy_Drone; 
  Truck_Index is a random integer in the range of 0 and NumberOfTrucks (can not take zero value and can take NumberOfTrucks value).
