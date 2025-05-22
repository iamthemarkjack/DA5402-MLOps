CREATE SCHEMA if not exists mta_data;
CREATE TABLE if not exists mta_data.mta_reports(
timestamp varchar NULL,
trip_id varchar NULL,
route_id varchar NULL,
trip_start_time varchar NULL,
trip_start_date varchar NULL,
vehicle_id varchar NULL,
vehicle_label varchar NULL,
vehicle_license_plate varchar NULL,
latitude varchar NULL,
longitude varchar NULL,
bearing varchar NULL,
speed varchar NULL,
stop_id varchar NULL,
stop_status varchar NULL,
occupancy_status varchar NULL,
congestion_level varchar NULL,
progress varchar NULL,
block_assigned varchar NULL,
dist_along_route varchar NULL,
dist_from_stop varchar NULL,
report_hour varchar NULL,  -- report_hour and report_date are created from the CSV newly in the script
report_date varchar NULL);

