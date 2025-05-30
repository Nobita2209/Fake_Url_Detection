import pandas as pd
from urllib.parse import urlparse
import ssl
import socket
import idna
df = pd.read_csv('C:/Users/DELL/Desktop/coding/project/Project.csv')

# Import necessary modules
import socket
import ssl
from urllib.parse import urlparse
import idna


# Define the checkSSL function
def checkSSL(url):
	try:
		# Parse the URL to extract the hostname
		parsed_url = urlparse(url)
		if parsed_url.scheme not in ['http', 'https']:
			raise ValueError("URL must start with 'http://' or 'https://'")
		hostname = parsed_url.hostname

		# Encode the hostname using idna
		hostname_idna = idna.encode(hostname)

		# Determine the port based on the scheme
		port = 443 if parsed_url.scheme == 'https' else 80

		# Create a socket
		s = socket.create_connection((hostname, port))

		# Create a SSL context
		context = ssl.create_default_context()

		# Wrap the socket with SSL
		with context.wrap_socket(s, server_hostname=hostname) as sock:
			# SSL connection established successfully
			return 1
	except ssl.SSLError as e:
		# SSL error occurred (self-signed certificate or other SSL-related errors)
		return 0
	except Exception as e:
		# Other errors occurred
		print(f"Error checking SSL certificate: {e}")
		return -1


# Example usage
url = "https://www.google.com/"
result = checkSSL(url)
print("SSL Check Result:", result)

#df['SSL Cert Check'] = df['url'].apply(lambda i: checkSSL(i))
#df.to_csv('C:/Users/DELL/Desktop/coding/project/Project.csv')