#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

using namespace std;

vector<vector<double>> extract_data_from_file(const string& filename);
bool is_current_feature(const vector<int>& current_features, int feature);
double find_accuracy(vector<vector<double>>& data, vector<int> feature_to_test);
void forward_selection(vector<vector<double>>& data);
void backward_elimination(vector<vector<double>>& data);

int main() {
    string filename;
    ifstream file;

    cout << "Welcome to Timmy Nguyen's Feature Selection Algorithm." << endl;

    // Prompts the user for a valid file name until a proper one is opened
    do {
        cout << "Type in the name of the file to test: ";
        cin >> filename;
        file.open(filename);
        if (!file.is_open()) {
            cout << "Invalid file. ";
        }
    } while (!file.is_open());

    file.close();

    // Extracts features and labels from the provided file and stores it in a vector
    vector<vector<double>> data = extract_data_from_file(filename);

    string choice;
    bool valid_choice = false;

    // Get basic stats for the introductory message
    int num_objects = data.size();           // Total rows in dataset
    int num_features = data[0].size() - 1;    // Total columns minus the label

    // Calculate baseline: Run with every feature included
    vector<int> all_features;
    for (int i = 1; i <= num_features; i++) {
        all_features.push_back(i);
    }

    double all_features_accuracy = find_accuracy(data, all_features);

    // Prompts the user for a valid choice until a proper one is provided
    do {
        cout << "\nType the number of the algorithm you want to run:" << endl;
        cout << "1) Forward Selection" << endl;
        cout << "2) Backward Elimination" << endl;

        cin >> choice;

        // Starts timer
        auto start_time = chrono::steady_clock::now();

        if (choice == "1") {
            cout << "\nThis dataset has " << num_features << " features (not including the class attribute), with " << num_objects << " instances." << endl;
            cout << "\nRunning nearest neighbor with all " << num_features << " features, using \"leaving-one-out\" evaluation, I get an accuracy of " << all_features_accuracy * 100 << "%" << endl;
            
            forward_selection(data);
            valid_choice = true;
        }
        else if (choice == "2") {
            cout << "\nThis dataset has " << num_features << " features (not including the class attribute), with " << num_objects << " instances." << endl;
            cout << "\nRunning nearest neighbor with all " << num_features << " features, using \"leaving-one-out\" evaluation, I get an accuracy of " << all_features_accuracy * 100 << "%" << endl;
            
            backward_elimination(data);
            valid_choice = true;
        }
        else {
            cout << "\nInvalid choice! Please enter exactly 1 or 2.\n" << endl;
            continue;
        }

        // Ends timer
        auto end_time = chrono::steady_clock::now();

        // Calculates elapsed time in seconds
        auto total_seconds = chrono::duration_cast<chrono::seconds>(end_time - start_time).count();

        // Conversion to hours and minutes
        long hours = total_seconds / 3600;
        long minutes = (total_seconds % 3600) / 60;
        long seconds = total_seconds % 60;

        cout << "\nTotal time elapsed: " << hours << " hours, " << minutes << " minutes, and " << seconds << " seconds." << endl;

    } while (!valid_choice);

    return 0;
}

// Function extracts data (features and labels) from the provided file
vector<vector<double>> extract_data_from_file(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line;

    // Gets the rows 1 by 1
    while (getline(file, line)) {
        vector<double> row;
        double value;
        stringstream ss(line);

        // Extracts each individual value (number) in each row then adds the row the vector
        while (ss >> value) {
            row.push_back(value);
        }

        // Prevents empty rows
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();

    return data;
}

// Function checks to see if the feature is already being used
bool is_current_feature(const vector<int>& current_features, int feature) {
    for (size_t i = 0; i < current_features.size(); i++) {
        if (current_features[i] == feature) return true;
    }
    return false;
}

// Function calculates the accuracy based on the provided features
// Leave-one-out-cross-validation occurs here. (Picks one feature at a time to test
// until it has picked all features)
double find_accuracy(vector<vector<double>>& data, vector<int> feature_to_test) {
    if (feature_to_test.empty()) return 0.0;

    int num_objects = data.size();
    int number_correctly_classified = 0;

    for (int i = 0; i < num_objects; i++) {
        int label_object_to_classify = (int)data[i][0];

        double nearest_neighbor_distance = HUGE_VAL; // HUGE_VAL to effectively represent infinity
        int nearest_neighbor_label = -1;

        for (int j = 0; j < num_objects; j++) {
            if (j == i) continue; // Prevents comparison with self (you would be the nearest to yourself)

            double sum = 0.0;

            // Calculates Euclidean Distance by ignoring features that aren't in the set being tested
            for (size_t feature_index = 0; feature_index < feature_to_test.size(); feature_index++) {
                int feature_num = feature_to_test[feature_index];
                double diff = data[i][feature_num] - data[j][feature_num];
                sum += diff * diff;
            }
            double distance = sqrt(sum);

            // Updates nearest neighbor distance and label if a closer neighbor is found
            if (distance < nearest_neighbor_distance) {
                nearest_neighbor_distance = distance;
                nearest_neighbor_label = (int)data[j][0];
            }
        }
        // Increment the number correctly classified by 1 if the nearest neighbor has the same label as the object
        if (label_object_to_classify == nearest_neighbor_label) {
            number_correctly_classified++;
        }
    }
    return (double)number_correctly_classified / num_objects;
}

/* For both searches: 
best_accuracy = FINAL best accuracy
best_accuracy_at_current_level = best accuracy with CURRENT number of features (i.e. current best accuracy for 2 features)
accuracy = accuracy just for ONE combination (i.e. a specific combination like {1, 3})
*/

// Forward Selection Search
// Starts off with no featueres then adds 1 at each level and keeps the best one.
void forward_selection(vector<vector<double>>& data) {
    int num_features = data[0].size() - 1;
    vector<int> current_features;
    vector<int> best_features;
    double best_accuracy = 0.0;

    cout << "\nBeginning search.\n" << endl;

    for (int i = 1; i <= num_features; i++) {
        int feature_to_add = -1;
        double best_accuracy_at_current_level = -1.0;

        // Adds every feature that isn't the current features
        for (int k = 1; k <= num_features; k++) {
            if (!is_current_feature(current_features, k)) {
                vector<int> features_to_test = current_features;
                features_to_test.push_back(k);

                double accuracy = find_accuracy(data, features_to_test);

                cout << "\tUsing feature(s) {";
                for (size_t j = 0; j < features_to_test.size(); j++) {
                    cout << features_to_test[j] << (j == features_to_test.size() - 1 ? "" : ",");
                }
                cout << "} accuracy is " << accuracy * 100 << "%" << endl;

                // Updates best accuracy at current level and best feature info if this feature combination is better
                if (accuracy > best_accuracy_at_current_level) {
                    best_accuracy_at_current_level = accuracy;
                    feature_to_add = k;
                }
            }
        }

        // Puts feature to be added onto the curent vector
        current_features.push_back(feature_to_add);
        cout << "\nFeature set {";
        for (size_t j = 0; j < current_features.size(); j++) {
            cout << current_features[j] << (j == current_features.size() - 1 ? "" : ",");
        }
        cout << "} was best, accuracy is " << best_accuracy_at_current_level * 100 << "%\n" << endl;

        // Updates best accuracy and best features if the best at the current level is better
        if (best_accuracy_at_current_level > best_accuracy) {
            best_accuracy = best_accuracy_at_current_level;
            best_features = current_features;
        }
        else {
            cout << "(Warning: Accuracy has decreased! Continuing search in case of local maxima)" << endl;
        }
    }

    cout << "\nFinished search!! The best feature subset is {";
    for (size_t j = 0; j < best_features.size(); j++) {
        cout << best_features[j] << (j == best_features.size() - 1 ? "" : ",");
    }
    cout << "}, which has an accuracy of " << best_accuracy * 100 << "%" << endl;
}

// Backward Elimination Search
// Everything works the same as Forward Selection except it starts with all
// features then remove the worst 1 at each level.
void backward_elimination(vector<vector<double>>& data) {
    int num_features = data[0].size() - 1;
    vector<int> current_features;

    // Starts with every feature
    for (int i = 1; i <= num_features; i++) current_features.push_back(i);

    double best_accuracy = find_accuracy(data, current_features);
    vector<int> best_features = current_features;

    cout << "\nBeginning search.\n" << endl;
    cout << "Full set accuracy: " << best_accuracy * 100 << "%\n" << endl;

    // Removes every feature in the current level
    for (int i = 1; i <= num_features; i++) {
        int feature_to_remove = -1;
        double best_accuracy_at_current_level = -1.0;

        for (int k = 1; k <= num_features; k++) {
            if (is_current_feature(current_features, k)) {
                vector<int> features_to_test;
                for (size_t j = 0; j < current_features.size(); j++) {
                    if (current_features[j] != k) features_to_test.push_back(current_features[j]);
                }

                double accuracy = find_accuracy(data, features_to_test);

                cout << "\tUsing feature(s) {";
                for (size_t j = 0; j < features_to_test.size(); j++) {
                    cout << features_to_test[j] << (j == features_to_test.size() - 1 ? "" : ",");
                }
                cout << "} accuracy is " << accuracy * 100 << "%" << endl;

                // Updates best accuracy at current level and best feature info if this feature combination is better
                if (accuracy > best_accuracy_at_current_level) {
                    best_accuracy_at_current_level = accuracy;
                    feature_to_remove = k;
                }
            }
        }

        // Removes the worst performing feature
        if (feature_to_remove != -1) {
            for (size_t j = 0; j < current_features.size(); j++) {
                if (current_features[j] == feature_to_remove) {
                    current_features.erase(current_features.begin() + j);
                    break;
                }
            }

            cout << "\nFeature set {";
            for (size_t j = 0; j < current_features.size(); j++) {
                cout << current_features[j] << (j == current_features.size() - 1 ? "" : ",");
            }
            cout << "} was best, accuracy is " << best_accuracy_at_current_level * 100 << "%\n" << endl;

            // Updates best accuracy and best features if the best at the current level is better
            if (best_accuracy_at_current_level > best_accuracy) {
                best_accuracy = best_accuracy_at_current_level;
                best_features = current_features;
            }
        }
    }

    cout << "\nFinished search!! The best feature subset is {";
    for (size_t j = 0; j < best_features.size(); j++) {
        cout << best_features[j] << (j == best_features.size() - 1 ? "" : ",");
    }

    cout << "}, which has an accuracy of " << best_accuracy * 100 << "%" << endl;
}
