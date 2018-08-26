using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace FeedbackPlatform.Common
{
    public class PasswordEncode
    {
       
            public string Hash(string value)
            {
                return Convert.ToBase64String(
                    System.Security.Cryptography.SHA256.Create()
                    .ComputeHash(System.Text.Encoding.UTF8.GetBytes(value))
                    );
            }
        
    }
}