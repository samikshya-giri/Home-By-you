import React, { useState, useEffect } from 'react';
import './AdminDashboard.css';

const AdminDashboard = () => {
    // State for login form
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [loginError, setLoginError] = useState('');

    // Dashboard states
    const [activeTab, setActiveTab] = useState('overview');
    const [metrics, setMetrics] = useState({
        accuracy: 0,
        precision: 0,
        recall: 0,
        f1Score: 0
    });
    const [classMetrics, setClassMetrics] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedImage, setSelectedImage] = useState(null);

    // Hardcoded credentials for demonstration
    const ADMIN_USERNAME = 'admin';
    const ADMIN_PASSWORD = 'admin';

    // Data from the classification report
    const classificationReport = {
        overall: {
            accuracy: 0.909,
            macro_precision: 0.810,
            macro_recall: 0.833,
            macro_f1: 0.815,
            weighted_precision: 0.913,
            weighted_recall: 0.909,
            weighted_f1: 0.909,
            total_support: 847
        },
        classes: [
            { className: 'bathtub', precision: 0.333, recall: 0.250, f1: 0.286, instances: 4 },
            { className: 'bed', precision: 0.955, recall: 0.955, f1: 0.955, instances: 44 },
            { className: 'bench', precision: 0.333, recall: 0.333, f1: 0.333, instances: 9 },
            { className: 'bookshelf', precision: 0.812, recall: 0.722, f1: 0.765, instances: 36 },
            { className: 'bunk_bed', precision: 0.857, recall: 1.000, f1: 0.923, instances: 6 },
            { className: 'cabinet', precision: 0.898, recall: 0.869, f1: 0.883, instances: 61 },
            { className: 'chair', precision: 0.948, recall: 0.958, f1: 0.953, instances: 191 },
            { className: 'couch', precision: 0.571, recall: 0.800, f1: 0.667, instances: 5 },
            { className: 'desk', precision: 0.837, recall: 1.000, f1: 0.911, instances: 36 },
            { className: 'dining_table', precision: 1.000, recall: 1.000, f1: 1.000, instances: 9 },
            { className: 'lamp', precision: 0.860, recall: 0.907, f1: 0.883, instances: 54 },
            { className: 'mirror', precision: 0.964, recall: 0.939, f1: 0.951, instances: 114 },
            { className: 'office_chair', precision: 0.833, recall: 0.833, f1: 0.833, instances: 6 },
            { className: 'plant', precision: 0.986, recall: 0.971, f1: 0.978, instances: 70 },
            { className: 'rug', precision: 0.844, recall: 0.844, f1: 0.844, instances: 32 },
            { className: 'sink', precision: 0.667, recall: 0.857, f1: 0.750, instances: 7 },
            { className: 'sofa', precision: 1.000, recall: 0.615, f1: 0.762, instances: 13 },
            { className: 'table', precision: 1.000, recall: 0.914, f1: 0.955, instances: 105 },
            { className: 'tv_stand', precision: 0.714, recall: 1.000, f1: 0.833, instances: 5 },
            { className: 'wardrobe', precision: 0.783, recall: 0.900, f1: 0.837, instances: 40 },
        ]
    };

    // Simulate API call to fetch data on component load
    useEffect(() => {
        // Only fetch data if the user is logged in
        if (isLoggedIn) {
            setLoading(true);
            setTimeout(() => {
                setMetrics({
                    accuracy: classificationReport.overall.accuracy,
                    precision: classificationReport.overall.weighted_precision,
                    recall: classificationReport.overall.weighted_recall,
                    f1Score: classificationReport.overall.weighted_f1,
                });
                setClassMetrics(classificationReport.classes);
                setLoading(false);
            }, 1500);
        }
    }, [isLoggedIn]);

    // Handle login form submission
    const handleLogin = (e) => {
        e.preventDefault();
        if (username === ADMIN_USERNAME && password === ADMIN_PASSWORD) {
            setIsLoggedIn(true);
            setLoginError('');
        } else {
            setLoginError('Invalid credentials. Please try again.');
        }
    };

    // Function to handle image click and show popup
    const handleImageClick = (src) => {
        setSelectedImage(src);
    };

    // Function to close the popup
    const handleClosePopup = () => {
        setSelectedImage(null);
    };

    const renderMetricCards = () => (
        <div className="metrics-grid">
            {Object.entries({
                Accuracy: { value: metrics.accuracy, icon: 'fas fa-bullseye' },
                'Weighted Precision': { value: metrics.precision, icon: 'fas fa-crosshairs' },
                'Weighted Recall': { value: metrics.recall, icon: 'fas fa-redo' },
                'Weighted F1 Score': { value: metrics.f1Score, icon: 'fas fa-chart-line' },
            }).map(([label, { value, icon }], index) => (
                <div key={index} className="metric-card">
                    <div className="metric-icon">
                        <i className={icon}></i>
                    </div>
                    <div className="metric-content">
                        <h3>{(value * 100).toFixed(1)}%</h3>
                        <p>{label}</p>
                    </div>
                    <div className="metric-progress">
                        <div
                            className="progress-bar"
                            style={{ width: `${value * 100}%` }}
                        ></div>
                    </div>
                </div>
            ))}
        </div>
    );

    const renderClassPerformance = () => (
        <div className="performance-table-container">
            <div className="table-header">
                <h3>Class Performance Metrics</h3>
                <button className="btn-sm">Export</button>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Instances</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {classMetrics.map((metric, index) => (
                        <tr key={index}>
                            <td>
                                <span className="class-badge">{metric.className}</span>
                            </td>
                            <td>
                                <div className="metric-value">
                                    <span>{(metric.precision * 100).toFixed(1)}%</span>
                                    <div className="value-bar">
                                        <div
                                            className="value-fill"
                                            style={{ width: `${metric.precision * 100}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </td>
                            <td>
                                <div className="metric-value">
                                    <span>{(metric.recall * 100).toFixed(1)}%</span>
                                    <div className="value-bar">
                                        <div
                                            className="value-fill"
                                            style={{ width: `${metric.recall * 100}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </td>
                            <td>
                                <div className="metric-value">
                                    <span>{(metric.f1 * 100).toFixed(1)}%</span>
                                    <div className="value-bar">
                                        <div
                                            className="value-fill"
                                            style={{ width: `${metric.f1 * 100}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </td>
                            <td>{metric.instances}</td>
                            <td>
                                <span className={`status-badge ${metric.f1 >= 0.9 ? 'status-good' : metric.f1 >= 0.8 ? 'status-warning' : 'status-bad'}`}>
                                    {metric.f1 >= 0.9 ? 'Excellent' : metric.f1 >= 0.8 ? 'Good' : 'Needs Improvement'}
                                </span>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );

    const renderModelVisualizations = () => (
        <div className="section-visualizations">
            <div className="visualizations-grid">
                <div className="visualization-card">
                    <h3>Confusion Matrix</h3>
                    <div className="img-container" onClick={() => handleImageClick('/plots/confusion_matrix_normalized.png')}>
                        <img
                            src="/plots/confusion_matrix_normalized.png"
                            alt="Confusion Matrix"
                            onError={(e) => {
                                e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxOCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgZmlsbD0iIzZjNzI4MCI+Q29uZnVzaW9uIE1hdHJpeCBJbWFnZTwvdGV4dD48L3N2Zz4=';
                            }}
                        />
                    </div>
                    <p>Normalized confusion matrix showing classification performance across all classes.</p>
                </div>
                <div className="visualization-card">
                    <h3>Feature Importance</h3>
                    <div className="img-container" onClick={() => handleImageClick('/plots/feature_importance.png')}>
                        <img
                            src="/plots/feature_importance.png"
                            alt="Feature Importance"
                            onError={(e) => {
                                e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxOCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgZmlsbD0iIzZjNzI4MCI+RmVhdHVyZSBJbXBvcnRhbmNlIEluZGljYXRvcnM8L3RleHQ+PC9zZ3Y+';
                            }}
                        />
                    </div>
                    <p>Relative importance of different features in the classification model.</p>
                </div>
                <div className="visualization-card">
                    <h3>Precision-Recall Curve</h3>
                    <div className="img-container" onClick={() => handleImageClick('/plots/precision_recall_curve.png')}>
                        <img
                            src="/plots/precision_recall_curve.png"
                            alt="Precision Recall Curve"
                            onError={(e) => {
                                e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxOCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgZmlsbD0iIzZjNzI4MCI+UHJlY2lzaW9uLVJlY2FsbCBDdXJ2ZTwvdGV4dD48L3N2Zz4=';
                            }}
                        />
                    </div>
                    <p>Precision-recall tradeoff for different classification thresholds.</p>
                </div>
            </div>
            <div className="distribution_classsection">
                <h3>Class Distribution</h3>
                <div className="img-container">
                    <div className="distribution-chart">
                        {classMetrics.map((metric, index) => (
                            <div key={index} className="distribution-bar-container">
                                <div className="distribution-label">{metric.className}</div>
                                <div className="distribution-bar">
                                    <div
                                        className="distribution-fill"
                                        style={{ width: `${(metric.instances / classificationReport.overall.total_support) * 100}%` }}
                                    ></div>
                                </div>
                                <div className="distribution-value">{metric.instances}</div>
                            </div>
                        ))}
                    </div>
                </div>
                <p>Distribution of instances across different furniture classes.</p>
            </div>
        </div>
    );

    const renderDashboardContent = () => (
        <>
            <header className="dashboard-header">
                <h1>AI Model Analytics Dashboard</h1>
                <p>Performance metrics and visualizations for your furniture classification model.</p>
            </header>
            <div className="dashboard-tabs">
                <button
                    className={activeTab === 'overview' ? 'tab-active' : ''}
                    onClick={() => setActiveTab('overview')}
                >
                    <i className="fas fa-chart-pie"></i> Overview
                </button>
                <button
                    className={activeTab === 'performance' ? 'tab-active' : ''}
                    onClick={() => setActiveTab('performance')}
                >
                    <i className="fas fa-tachometer-alt"></i> Performance
                </button>
                <button
                    className={activeTab === 'visualizations' ? 'tab-active' : ''}
                    onClick={() => setActiveTab('visualizations')}
                >
                    <i className="fas fa-chart-bar"></i> Visualizations
                </button>
            </div>
            {loading ? (
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <p>Loading model analytics...</p>
                </div>
            ) : (
                <div className="dashboard-content">
                    {activeTab === 'overview' && (
                        <>
                            {renderMetricCards()}
                            <div className="overview-grid">
                                <div className="overview-card">
                                    <h3>Model Summary</h3>
                                    <div className="summary-stats">
                                        <div className="summary-stat">
                                            <span className="stat-value">{classificationReport.overall.total_support}</span>
                                            <span className="stat-label">Total Predictions</span>
                                        </div>
                                        <div className="summary-stat">
                                            <span className="stat-value">{classificationReport.classes.length}</span>
                                            <span className="stat-label">Furniture Classes</span>
                                        </div>
                                        <div className="summary-stat">
                                            <span className="stat-value">{(classificationReport.overall.accuracy * 100).toFixed(1)}%</span>
                                            <span className="stat-label">Overall Accuracy</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="overview-card">
                                    <h3>Recent Activity</h3>
                                    <div className="activity-list">
                                        <div className="activity-item">
                                            <div className="activity-icon success">
                                                <i className="fas fa-check-circle"></i>
                                            </div>
                                            <div className="activity-content">
                                                <p>Performance alert: Chair classification dropped 5% </p>
                                                <span className="activity-time">2 days ago</span>
                                            </div>
                                        </div>
                                        <div className="activity-item">
                                            <div className="activity-icon info">
                                                <i className="fas fa-info-circle"></i>
                                            </div>
                                            <div className="activity-content">
                                                <p>New dataset version uploaded</p>
                                                <span className="activity-time">5 days ago</span>
                                            </div>
                                        </div>
                                        <div className="activity-item">
                                            <div className="activity-icon warning">
                                                <i className="fas fa-exclamation-triangle"></i>
                                            </div>
                                            <div className="activity-content">
                                                <p> Model retraining completed</p>
                                                <span className="activity-time">15 days ago</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}
                    {activeTab === 'performance' && renderClassPerformance()}
                    {activeTab === 'visualizations' && renderModelVisualizations()}
                </div>
            )}
            {selectedImage && (
                <div className="image-popup-overlay" onClick={handleClosePopup}>
                    <div className="image-popup-content">
                        <span className="image-popup-close" onClick={handleClosePopup}>&times;</span>
                        <img src={selectedImage} alt="Full-size visualization" />
                    </div>
                </div>
            )}
        </>
    );

    const renderLoginForm = () => (
        <div className="login-container">
            <form className="login-form" onSubmit={handleLogin}>
                <h2>Admin Login</h2>
                {loginError && <p className="login-error">{loginError}</p>}
                <div className="form-group">
                    <label htmlFor="username">Username</label>
                    <input
                        type="text"
                        id="username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="password">Password</label>
                    <input
                        type="password"
                        id="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                </div>
                <button type="submit" className="login-btn">Log In</button>
            </form>
        </div>
    );

    return (
        <div className="admin-dashboard">
            {isLoggedIn ? renderDashboardContent() : renderLoginForm()}
        </div>
    );
};

export default AdminDashboard;