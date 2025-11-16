import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import hashlib

# --------------------------
# Core Modules
# --------------------------

class FundManagementSystem:
    def __init__(self):
        self.initialize_session_state()
        self.setup_ui()
        
    def initialize_session_state(self):
        """Initialize all required session state variables"""
        if 'funds' not in st.session_state:
            st.session_state.funds = pd.DataFrame(columns=[
                'FundID', 'FundName', 'StartDate', 'EndDate', 'TotalAmount', 
                'Allocated', 'Utilized', 'Remaining', 'Status'
            ])
            
        if 'benefit_plans' not in st.session_state:
            st.session_state.benefit_plans = pd.DataFrame(columns=[
                'PlanID', 'PlanName', 'FundID', 'Deductible', 'OOPMax', 
                'Coinsurance', 'Copay', 'CoveredServices'
            ])
            
        if 'claims' not in st.session_state:
            st.session_state.claims = pd.DataFrame(columns=[
                'ClaimID', 'MemberID', 'PlanID', 'ProviderID', 'ServiceDate',
                'ServiceType', 'AmountBilled', 'AmountApproved', 'Status', 
                'ProcessedDate'
            ])
            
        if 'members' not in st.session_state:
            st.session_state.members = pd.DataFrame(columns=[
                'MemberID', 'FirstName', 'LastName', 'DOB', 'EnrollmentDate',
                'PlanID', 'Status'
            ])
            
        if 'providers' not in st.session_state:
            st.session_state.providers = pd.DataFrame(columns=[
                'ProviderID', 'Name', 'Specialty', 'NetworkStatus', 
                'ContractRate', 'EffectiveDate'
            ])
    
    def setup_ui(self):
        """Set up the main user interface"""
        st.set_page_config(layout="wide", page_title="Fund Management System")
        
        # Authentication (to align with existing system)
        if not self.authenticate_user():
            return
            
        # Main navigation
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.selectbox("Choose a module", [
            "Dashboard", 
            "Fund Management", 
            "Benefit Design", 
            "Claims Processing", 
            "Provider Network",
            "Reports & Analytics"
        ])
        
        # Route to selected module
        if app_mode == "Dashboard":
            self.show_dashboard()
        elif app_mode == "Fund Management":
            self.fund_management()
        elif app_mode == "Benefit Design":
            self.benefit_design()
        elif app_mode == "Claims Processing":
            self.claims_processing()
        elif app_mode == "Provider Network":
            self.provider_network()
        elif app_mode == "Reports & Analytics":
            self.reports_analytics()
    
    # --------------------------
    # Authentication (aligned with existing system)
    # --------------------------
    def authenticate_user(self):
        """User authentication aligned with existing system"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            
        if not st.session_state.authenticated:
            st.title("Fund Management System Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if self.validate_credentials(username, password):
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")
            return False
        return True
    
    def validate_credentials(self, username, password):
        """Simplified authentication - to be replaced with your existing auth system"""
        # This is a placeholder - integrate with your existing authentication
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        return username == "admin" and hashed_pw == "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"
    
    # --------------------------
    # Core Functionality Modules
    # --------------------------
    
    def show_dashboard(self):
        """Main dashboard with fund overview and key metrics"""
        st.title("Fund Management Dashboard")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Funds", f"${st.session_state.funds['TotalAmount'].sum():,.2f}")
        with col2:
            st.metric("Active Members", len(st.session_state.members[st.session_state.members['Status'] == 'Active']))
        with col3:
            utilization = st.session_state.funds['Utilized'].sum() / st.session_state.funds['TotalAmount'].sum() * 100
            st.metric("Fund Utilization", f"{utilization:.1f}%")
        
        # Fund status visualization
        st.subheader("Fund Status Overview")
        if not st.session_state.funds.empty:
            fig = px.pie(st.session_state.funds, names='Status', values='TotalAmount',
                         title='Fund Allocation by Status')
            st.plotly_chart(fig, use_container_width=True)
            
            # Time-based fund utilization
            st.subheader("Monthly Fund Utilization")
            # Simulate some time-series data for demo
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='M')
            utilization_data = pd.DataFrame({
                'Month': dates,
                'AmountUtilized': np.random.randint(50000, 200000, size=len(dates)),
                'AmountAllocated': np.random.randint(100000, 300000, size=len(dates))
            })
            fig = px.line(utilization_data, x='Month', y=['AmountUtilized', 'AmountAllocated'],
                          title='Monthly Fund Utilization vs Allocation')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No fund data available. Please set up funds in the Fund Management module.")
    
    def fund_management(self):
        """Manage fund setup and allocations"""
        st.title("Fund Management")
        
        tab1, tab2, tab3 = st.tabs(["Create Fund", "View/Edit Funds", "Allocate Funds"])
        
        with tab1:
            with st.form("create_fund"):
                st.subheader("Create New Fund")
                fund_name = st.text_input("Fund Name")
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")
                total_amount = st.number_input("Total Amount ($)", min_value=0.0, step=1000.0)
                
                if st.form_submit_button("Create Fund"):
                    new_fund = {
                        'FundID': f"FUND-{len(st.session_state.funds)+1:04d}",
                        'FundName': fund_name,
                        'StartDate': start_date,
                        'EndDate': end_date,
                        'TotalAmount': total_amount,
                        'Allocated': 0.0,
                        'Utilized': 0.0,
                        'Remaining': total_amount,
                        'Status': 'Active'
                    }
                    st.session_state.funds = st.session_state.funds.append(new_fund, ignore_index=True)
                    st.success("Fund created successfully!")
        
        with tab2:
            st.subheader("Existing Funds")
            if not st.session_state.funds.empty:
                edited_df = st.data_editor(st.session_state.funds)
                if st.button("Save Changes"):
                    st.session_state.funds = edited_df
                    st.success("Changes saved!")
            else:
                st.warning("No funds available. Please create a fund first.")
        
        with tab3:
            st.subheader("Allocate Funds to Benefit Plans")
            if not st.session_state.funds.empty and not st.session_state.benefit_plans.empty:
                fund_id = st.selectbox("Select Fund", st.session_state.funds['FundID'])
                plan_id = st.selectbox("Select Benefit Plan", st.session_state.benefit_plans['PlanID'])
                amount = st.number_input("Allocation Amount ($)", min_value=0.0)
                
                if st.button("Allocate Funds"):
                    fund_idx = st.session_state.funds[st.session_state.funds['FundID'] == fund_id].index[0]
                    if amount <= st.session_state.funds.loc[fund_idx, 'Remaining']:
                        st.session_state.funds.loc[fund_idx, 'Allocated'] += amount
                        st.session_state.funds.loc[fund_idx, 'Remaining'] -= amount
                        
                        # Update benefit plan with fund reference
                        plan_idx = st.session_state.benefit_plans[st.session_state.benefit_plans['PlanID'] == plan_id].index[0]
                        st.session_state.benefit_plans.loc[plan_idx, 'FundID'] = fund_id
                        
                        st.success(f"Allocated ${amount:,.2f} from {fund_id} to {plan_id}")
                    else:
                        st.error("Insufficient remaining funds for this allocation")
            else:
                st.warning("Please ensure you have both funds and benefit plans created.")
    
    def benefit_design(self):
        """Custom benefit plan configuration"""
        st.title("Benefit Plan Design")
        
        tab1, tab2 = st.tabs(["Create Plan", "View/Edit Plans"])
        
        with tab1:
            with st.form("create_plan"):
                st.subheader("Create New Benefit Plan")
                plan_name = st.text_input("Plan Name")
                deductible = st.number_input("Deductible ($)", min_value=0.0)
                oop_max = st.number_input("Out-of-Pocket Maximum ($)", min_value=0.0)
                coinsurance = st.slider("Coinsurance (%)", 0, 100, 20)
                copay = st.number_input("Copay ($)", min_value=0.0)
                covered_services = st.multiselect("Covered Services", [
                    "Preventive Care", "Primary Care", "Specialist Visits",
                    "Emergency Care", "Hospitalization", "Prescription Drugs",
                    "Mental Health", "Physical Therapy"
                ])
                
                if st.form_submit_button("Create Plan"):
                    new_plan = {
                        'PlanID': f"PLAN-{len(st.session_state.benefit_plans)+1:04d}",
                        'PlanName': plan_name,
                        'FundID': "",
                        'Deductible': deductible,
                        'OOPMax': oop_max,
                        'Coinsurance': coinsurance,
                        'Copay': copay,
                        'CoveredServices': ", ".join(covered_services)
                    }
                    st.session_state.benefit_plans = st.session_state.benefit_plans.append(new_plan, ignore_index=True)
                    st.success("Benefit plan created successfully!")
        
        with tab2:
            st.subheader("Existing Benefit Plans")
            if not st.session_state.benefit_plans.empty:
                edited_df = st.data_editor(st.session_state.benefit_plans)
                if st.button("Save Changes"):
                    st.session_state.benefit_plans = edited_df
                    st.success("Changes saved!")
            else:
                st.warning("No benefit plans available. Please create a plan first.")
    
    def claims_processing(self):
        """Claims processing and simulation"""
        st.title("Claims Processing")
        
        tab1, tab2, tab3 = st.tabs(["Submit Claim", "Process Claims", "Claims History"])
        
        with tab1:
            with st.form("submit_claim"):
                st.subheader("Submit New Claim")
                member_id = st.selectbox("Member ID", st.session_state.members['MemberID'])
                provider_id = st.selectbox("Provider ID", st.session_state.providers['ProviderID'])
                service_date = st.date_input("Service Date")
                service_type = st.selectbox("Service Type", [
                    "Preventive Care", "Primary Care", "Specialist Visit",
                    "Emergency Care", "Hospitalization", "Prescription"
                ])
                amount_billed = st.number_input("Amount Billed ($)", min_value=0.0)
                
                if st.form_submit_button("Submit Claim"):
                    member_plan = st.session_state.members[st.session_state.members['MemberID'] == member_id]['PlanID'].values[0]
                    new_claim = {
                        'ClaimID': f"CLM-{len(st.session_state.claims)+1:06d}",
                        'MemberID': member_id,
                        'PlanID': member_plan,
                        'ProviderID': provider_id,
                        'ServiceDate': service_date,
                        'ServiceType': service_type,
                        'AmountBilled': amount_billed,
                        'AmountApproved': 0.0,  # To be set during processing
                        'Status': "Pending",
                        'ProcessedDate': None
                    }
                    st.session_state.claims = st.session_state.claims.append(new_claim, ignore_index=True)
                    st.success("Claim submitted successfully!")
        
        with tab2:
            st.subheader("Pending Claims")
            pending_claims = st.session_state.claims[st.session_state.claims['Status'] == "Pending"]
            
            if not pending_claims.empty:
                for _, claim in pending_claims.iterrows():
                    with st.expander(f"Claim {claim['ClaimID']} - {claim['ServiceType']}"):
                        st.write(f"Member: {claim['MemberID']}")
                        st.write(f"Provider: {claim['ProviderID']}")
                        st.write(f"Service Date: {claim['ServiceDate']}")
                        st.write(f"Amount Billed: ${claim['AmountBilled']:,.2f}")
                        
                        # Get member's plan details
                        plan_details = st.session_state.benefit_plans[
                            st.session_state.benefit_plans['PlanID'] == claim['PlanID']
                        ].iloc[0]
                        
                        # Simplified claims adjudication logic
                        approved_amount = min(claim['AmountBilled'], claim['AmountBilled'] * 0.8)  # Example: approve 80%
                        
                        st.write(f"Plan Deductible: ${plan_details['Deductible']:,.2f}")
                        st.write(f"Plan Coinsurance: {plan_details['Coinsurance']}%")
                        
                        if st.button(f"Approve Claim {claim['ClaimID']}", key=f"approve_{claim['ClaimID']}"):
                            # Update claim status
                            claim_idx = st.session_state.claims[st.session_state.claims['ClaimID'] == claim['ClaimID']].index[0]
                            st.session_state.claims.loc[claim_idx, 'AmountApproved'] = approved_amount
                            st.session_state.claims.loc[claim_idx, 'Status'] = "Approved"
                            st.session_state.claims.loc[claim_idx, 'ProcessedDate'] = datetime.now().date()
                            
                            # Update fund utilization
                            fund_id = plan_details['FundID']
                            if fund_id:
                                fund_idx = st.session_state.funds[st.session_state.funds['FundID'] == fund_id].index[0]
                                st.session_state.funds.loc[fund_idx, 'Utilized'] += approved_amount
                                st.session_state.funds.loc[fund_idx, 'Remaining'] -= approved_amount
                            
                            st.experimental_rerun()
                        
                        if st.button(f"Deny Claim {claim['ClaimID']}", key=f"deny_{claim['ClaimID']}"):
                            claim_idx = st.session_state.claims[st.session_state.claims['ClaimID'] == claim['ClaimID']].index[0]
                            st.session_state.claims.loc[claim_idx, 'Status'] = "Denied"
                            st.session_state.claims.loc[claim_idx, 'ProcessedDate'] = datetime.now().date()
                            st.experimental_rerun()
            else:
                st.info("No pending claims to process")
        
        with tab3:
            st.subheader("Claims History")
            if not st.session_state.claims.empty:
                st.dataframe(st.session_state.claims)
            else:
                st.warning("No claims history available")
    
    def provider_network(self):
        """Provider network management"""
        st.title("Provider Network Management")
        
        tab1, tab2 = st.tabs(["Add Provider", "View/Edit Providers"])
        
        with tab1:
            with st.form("add_provider"):
                st.subheader("Add New Provider")
                name = st.text_input("Provider Name")
                specialty = st.selectbox("Specialty", [
                    "Primary Care", "Cardiology", "Dermatology", 
                    "Endocrinology", "Gastroenterology", "Neurology",
                    "Oncology", "Orthopedics", "Pediatrics", "Psychiatry"
                ])
                network_status = st.selectbox("Network Status", ["In-Network", "Out-of-Network"])
                contract_rate = st.slider("Contract Rate (%)", 0, 100, 80)
                effective_date = st.date_input("Effective Date")
                
                if st.form_submit_button("Add Provider"):
                    new_provider = {
                        'ProviderID': f"PROV-{len(st.session_state.providers)+1:04d}",
                        'Name': name,
                        'Specialty': specialty,
                        'NetworkStatus': network_status,
                        'ContractRate': contract_rate,
                        'EffectiveDate': effective_date
                    }
                    st.session_state.providers = st.session_state.providers.append(new_provider, ignore_index=True)
                    st.success("Provider added successfully!")
        
        with tab2:
            st.subheader("Provider Directory")
            if not st.session_state.providers.empty:
                edited_df = st.data_editor(st.session_state.providers)
                if st.button("Save Changes"):
                    st.session_state.providers = edited_df
                    st.success("Changes saved!")
            else:
                st.warning("No providers available. Please add providers first.")
    
    def reports_analytics(self):
        """Reporting and analytics module"""
        st.title("Reports & Analytics")
        
        report_type = st.selectbox("Select Report", [
            "Fund Utilization Summary",
            "Claims Analysis",
            "Member Enrollment",
            "Provider Performance",
            "Risk Analysis"
        ])
        
        if report_type == "Fund Utilization Summary":
            st.subheader("Fund Utilization Summary")
            if not st.session_state.funds.empty:
                fig = px.bar(st.session_state.funds, 
                             x='FundName', 
                             y=['Allocated', 'Utilized', 'Remaining'],
                             barmode='group',
                             title='Fund Allocation vs Utilization')
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(st.session_state.funds)
            else:
                st.warning("No fund data available")
        
        elif report_type == "Claims Analysis":
            st.subheader("Claims Analysis")
            if not st.session_state.claims.empty:
                # Claims by type
                fig1 = px.pie(st.session_state.claims, 
                              names='ServiceType', 
                              title='Claims by Service Type')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Claims status
                fig2 = px.histogram(st.session_state.claims, 
                                   x='Status', 
                                   color='Status',
                                   title='Claims by Status')
                st.plotly_chart(fig2, use_container_width=True)
                
                # Time series of claims
                claims_by_month = st.session_state.claims.copy()
                claims_by_month['Month'] = pd.to_datetime(claims_by_month['ServiceDate']).dt.to_period('M')
                claims_by_month = claims_by_month.groupby('Month').agg({
                    'AmountBilled': 'sum',
                    'AmountApproved': 'sum'
                }).reset_index()
                claims_by_month['Month'] = claims_by_month['Month'].astype(str)
                
                fig3 = px.line(claims_by_month, 
                              x='Month', 
                              y=['AmountBilled', 'AmountApproved'],
                              title='Monthly Claims Amounts')
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("No claims data available")
        
        elif report_type == "Member Enrollment":
            st.subheader("Member Enrollment")
            if not st.session_state.members.empty:
                # Enrollment by plan
                fig = px.pie(st.session_state.members, 
                             names='PlanID', 
                             title='Member Distribution by Plan')
                st.plotly_chart(fig, use_container_width=True)
                
                # Status distribution
                fig2 = px.histogram(st.session_state.members, 
                                   x='Status', 
                                   color='Status',
                                   title='Member Status Distribution')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("No member data available")
        
        elif report_type == "Provider Performance":
            st.subheader("Provider Performance")
            if not st.session_state.providers.empty and not st.session_state.claims.empty:
                # Merge claims with providers
                provider_claims = pd.merge(
                    st.session_state.claims,
                    st.session_state.providers,
                    on='ProviderID'
                )
                
                # Top providers by claim volume
                top_providers = provider_claims['Name'].value_counts().reset_index()
                top_providers.columns = ['Provider', 'ClaimCount']
                
                fig1 = px.bar(top_providers.head(10), 
                             x='Provider', 
                             y='ClaimCount',
                             title='Top Providers by Claim Volume')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Approval rates by provider
                provider_stats = provider_claims.groupby(['Name', 'Status']).size().unstack().fillna(0)
                provider_stats['ApprovalRate'] = provider_stats.get('Approved', 0) / provider_stats.sum(axis=1) * 100
                
                fig2 = px.bar(provider_stats.reset_index(), 
                             x='Name', 
                             y='ApprovalRate',
                             title='Approval Rates by Provider')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Insufficient data for provider performance analysis")
        
        elif report_type == "Risk Analysis":
            st.subheader("Risk Analysis")
            st.write("This section would integrate with the existing Insurance Loss Ratio Prediction system")
            
            # Placeholder for integration with loss ratio prediction
            if st.button("Run Risk Analysis"):
                # This would call the existing loss ratio prediction system
                st.info("Integration with Insurance Loss Ratio Prediction system would happen here")
                
                # Simulated risk analysis results
                risk_data = pd.DataFrame({
                    'RiskFactor': ['Claims Volatility', 'High-Cost Claims', 'Provider Concentration', 'Member Demographics'],
                    'Score': [3.2, 4.1, 2.8, 3.5],
                    'Status': ['Moderate', 'High', 'Low', 'Moderate']
                })
                
                fig = px.bar(risk_data, 
                             x='RiskFactor', 
                             y='Score',
                             color='Status',
                             title='Risk Factor Analysis')
                st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Main Application Execution
# --------------------------

if __name__ == "__main__":
    app = FundManagementSystem()